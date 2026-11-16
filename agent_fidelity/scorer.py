"""Bayesian fidelity scoring.

Model
-----
Let theta be the probability that one closed-loop turn of the agent is
physically faithful, i.e. breaks none of the configured constraints. We hold a
Beta(alpha, beta) belief over theta and update it after every turn:

* a clean turn adds 1 to alpha;
* a turn with violations adds, for each violated constraint,
  ``weight * (1 + min(deviation, deviation_cap))`` to beta, so more and worse
  violations count as stronger evidence.

Before each update the evidence gathered so far is shrunk toward the prior by
``discount`` (exponential forgetting). Without it an agent with a long clean
history could misbehave for a long time before the posterior moved; with it
the belief tracks *recent* behaviour, which is what makes the trust curve
decay when an agent starts to drift.

The reported trust is the posterior mean, and the "confidence interval" is the
equal-tailed Beta credible interval at ``credible_level``.

Risk
----
Any constraint violation raises a HIGH alert on the spot; it is CRITICAL when
the posterior mean has also fallen below ``trust_threshold``. With no violation
this turn the level is ELEVATED while the lower credible bound is still under
the threshold, otherwise NOMINAL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Sequence

from scipy.stats import beta as beta_dist

from .constraints import ConstraintResult, PhysicalConstraint
from .trace import Trace, Turn

logger = logging.getLogger("agent_fidelity")


class RiskLevel(IntEnum):
    NOMINAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Assessment:
    """Posterior belief about the agent after one turn."""

    turn: int
    alpha: float
    beta: float
    trust: float  # posterior mean of theta
    ci_low: float
    ci_high: float
    p_below_threshold: float  # P(theta < trust_threshold)
    risk: RiskLevel
    results: list[ConstraintResult] = field(default_factory=list)

    @property
    def violations(self) -> list[ConstraintResult]:
        return [r for r in self.results if not r.satisfied]

    def to_record(self) -> dict[str, Any]:
        """Flat row, convenient for a DataFrame."""
        return {
            "turn": self.turn,
            "trust": self.trust,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_below_threshold": self.p_below_threshold,
            "risk": self.risk.name,
            "violations": len(self.violations),
            "violated": "; ".join(r.constraint for r in self.violations),
        }


@dataclass
class Alert:
    turn: int
    level: RiskLevel
    message: str
    trust: float
    ci_low: float
    ci_high: float
    violations: list[ConstraintResult] = field(default_factory=list)


AlertHandler = Callable[[Alert], None]


class BayesianScorer:
    def __init__(
        self,
        constraints: Sequence[PhysicalConstraint],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        discount: float = 0.95,
        credible_level: float = 0.95,
        trust_threshold: float = 0.6,
        deviation_cap: float = 2.0,
        on_alert: AlertHandler | None = None,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior parameters must be positive")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if not 0 < credible_level < 1:
            raise ValueError("credible_level must be in (0, 1)")
        if not 0 < trust_threshold < 1:
            raise ValueError("trust_threshold must be in (0, 1)")

        self.constraints = list(constraints)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.discount = discount
        self.credible_level = credible_level
        self.trust_threshold = trust_threshold
        self.deviation_cap = deviation_cap
        self._handlers: list[AlertHandler] = [on_alert] if on_alert else []
        self.reset()

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.history: list[Assessment] = []
        self.alerts: list[Alert] = []
        self._previous: Turn | None = None

    def on_alert(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    @property
    def trust(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level: float | None = None) -> tuple[float, float]:
        level = self.credible_level if level is None else level
        tail = (1.0 - level) / 2.0
        lo, hi = beta_dist.ppf([tail, 1.0 - tail], self.alpha, self.beta)
        return float(lo), float(hi)

    # -- scoring -----------------------------------------------------------

    def score_turn(self, turn: Turn) -> Assessment:
        """Check one turn against every constraint and update the posterior."""
        results = [
            r
            for c in self.constraints
            if (r := c.evaluate(turn, self._previous)) is not None
        ]
        violations = [r for r in results if not r.satisfied]

        if results:  # a turn with nothing to check is not evidence either way
            g = self.discount
            self.alpha = self.prior_alpha + g * (self.alpha - self.prior_alpha)
            self.beta = self.prior_beta + g * (self.beta - self.prior_beta)
            if violations:
                self.beta += sum(
                    r.weight * (1.0 + min(r.deviation, self.deviation_cap))
                    for r in violations
                )
            else:
                self.alpha += 1.0

        lo, hi = self.credible_interval()
        trust = self.trust
        if violations:
            risk = RiskLevel.CRITICAL if trust < self.trust_threshold else RiskLevel.HIGH
        elif lo < self.trust_threshold:
            risk = RiskLevel.ELEVATED
        else:
            risk = RiskLevel.NOMINAL

        assessment = Assessment(
            turn=turn.index,
            alpha=self.alpha,
            beta=self.beta,
            trust=trust,
            ci_low=lo,
            ci_high=hi,
            p_below_threshold=float(
                beta_dist.cdf(self.trust_threshold, self.alpha, self.beta)
            ),
            risk=risk,
            results=results,
        )
        self.history.append(assessment)
        self._previous = turn

        if violations:
            self._raise_alert(assessment)
        return assessment

    def observe(self, trace: Trace) -> list[Assessment]:
        """Score every closed turn of ``trace`` that has not been scored yet."""
        done = self.history[-1].turn if self.history else -1
        return [self.score_turn(t) for t in trace.turns() if t.index > done]

    def attach(self, trace: Trace) -> None:
        """Score each turn of ``trace`` automatically as soon as it closes."""
        trace.on_turn_end(self.score_turn)

    def records(self) -> list[dict[str, Any]]:
        """The trust curve as flat rows (one per scored turn)."""
        return [a.to_record() for a in self.history]

    # -- alerts ------------------------------------------------------------

    def _raise_alert(self, a: Assessment) -> None:
        detail = "; ".join(r.message or r.constraint for r in a.violations)
        alert = Alert(
            turn=a.turn,
            level=a.risk,
            message=f"{a.risk.name} RISK at turn {a.turn}: {detail}",
            trust=a.trust,
            ci_low=a.ci_low,
            ci_high=a.ci_high,
            violations=a.violations,
        )
        self.alerts.append(alert)
        logger.warning(
            "%s (trust %.2f, %d%% CI %.2f-%.2f)",
            alert.message, a.trust, round(self.credible_level * 100), a.ci_low, a.ci_high,
        )
        for handler in self._handlers:
            handler(alert)
