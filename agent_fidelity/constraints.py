"""Physical constraints an agent's behaviour is checked against.

Each constraint looks at one :class:`~agent_fidelity.trace.Turn` (and, for
rate limits, the turn before it) and returns a :class:`ConstraintResult`, or
``None`` when the turn carries no data the constraint applies to.

``deviation`` is how far outside the limit the value landed, normalised by the
size of the limit itself, so 0.5 means "50% of the allowed span too far". The
scorer uses it to weigh bad violations more heavily than marginal ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .trace import Turn


@dataclass
class ConstraintResult:
    constraint: str
    satisfied: bool
    deviation: float = 0.0
    message: str = ""
    value: Any = None
    weight: float = 1.0


class PhysicalConstraint(Protocol):
    name: str
    weight: float

    def evaluate(self, turn: Turn, previous: Turn | None = None) -> ConstraintResult | None: ...


def _number(values: dict[str, Any], key: str) -> float | None:
    v = values.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


@dataclass
class RangeConstraint:
    """``low <= value <= high`` for one quantity.

    ``source`` picks where the value is read from: ``"observed"`` (physical
    feedback), ``"action"`` (tool-call arguments) or ``"expected"`` (the
    agent's own prediction).
    """

    key: str
    low: float
    high: float
    source: str = "observed"
    weight: float = 1.0
    name: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must be <= high")
        self.name = self.name or f"{self.source}.{self.key} in [{self.low:g}, {self.high:g}]"

    def evaluate(self, turn: Turn, previous: Turn | None = None) -> ConstraintResult | None:
        v = _number(turn.values(self.source), self.key)
        if v is None:
            return None
        excess = max(self.low - v, v - self.high, 0.0)
        span = (self.high - self.low) or 1.0
        ok = excess == 0.0
        msg = "" if ok else (
            f"{self.key} = {v:.3g}{self.unit} is outside "
            f"[{self.low:g}, {self.high:g}]{self.unit}"
        )
        return ConstraintResult(self.name, ok, excess / span, msg, v, self.weight)


@dataclass
class RateConstraint:
    """Limits how fast a quantity may change between consecutive turns."""

    key: str
    max_delta: float
    source: str = "observed"
    weight: float = 1.0
    name: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.max_delta <= 0:
            raise ValueError("max_delta must be positive")
        self.name = self.name or f"|d {self.source}.{self.key}| <= {self.max_delta:g}"

    def evaluate(self, turn: Turn, previous: Turn | None = None) -> ConstraintResult | None:
        if previous is None:
            return None
        now = _number(turn.values(self.source), self.key)
        before = _number(previous.values(self.source), self.key)
        if now is None or before is None:
            return None
        delta = abs(now - before)
        excess = max(delta - self.max_delta, 0.0)
        ok = excess == 0.0
        msg = "" if ok else (
            f"{self.key} changed by {delta:.3g}{self.unit} in one turn "
            f"(limit {self.max_delta:g}{self.unit})"
        )
        return ConstraintResult(self.name, ok, excess / self.max_delta, msg, delta, self.weight)


@dataclass
class ConsistencyConstraint:
    """The agent's predicted outcome must match physical feedback within
    ``tolerance``. This is the direct test of whether its world model has
    drifted away from reality."""

    key: str
    tolerance: float
    weight: float = 1.0
    name: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.name = self.name or f"|expected - observed {self.key}| <= {self.tolerance:g}"

    def evaluate(self, turn: Turn, previous: Turn | None = None) -> ConstraintResult | None:
        expected = _number(turn.expected, self.key)
        observed = _number(turn.observed, self.key)
        if expected is None or observed is None:
            return None
        gap = abs(expected - observed)
        excess = max(gap - self.tolerance, 0.0)
        ok = excess == 0.0
        msg = "" if ok else (
            f"agent expected {self.key} = {expected:.3g}{self.unit} but physics "
            f"returned {observed:.3g}{self.unit} (tolerance {self.tolerance:g}{self.unit})"
        )
        return ConstraintResult(self.name, ok, excess / self.tolerance, msg, gap, self.weight)


@dataclass
class PredicateConstraint:
    """Escape hatch: any ``fn(turn, previous) -> bool | None``.

    Return ``None`` when the turn has nothing to check."""

    name: str
    fn: Callable[[Turn, Turn | None], bool | None]
    weight: float = 1.0
    message: str = ""

    def evaluate(self, turn: Turn, previous: Turn | None = None) -> ConstraintResult | None:
        ok = self.fn(turn, previous)
        if ok is None:
            return None
        return ConstraintResult(
            self.name,
            bool(ok),
            0.0 if ok else 1.0,
            "" if ok else (self.message or f"{self.name} failed"),
            None,
            self.weight,
        )
