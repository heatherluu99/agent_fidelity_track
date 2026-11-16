import pytest

from agent_fidelity import (
    BayesianScorer,
    ConsistencyConstraint,
    PredicateConstraint,
    RangeConstraint,
    RateConstraint,
    RiskLevel,
    Trace,
)
from agent_fidelity.simulate import ThermalLoopSim


def turn_with(trace: Trace, power=50.0, expected=60.0, observed=60.0):
    with trace.turn():
        trace.tool_call("set_heater", {"power": power}, expected={"temp": expected})
        trace.feedback({"temp": observed})
    return trace.get_turn(trace.current_turn - 1)


CONSTRAINTS = [
    RangeConstraint("power", 0, 100, source="action"),
    RangeConstraint("temp", 20, 70),
    RateConstraint("temp", 3.0),
    ConsistencyConstraint("temp", 2.0),
]


def test_prior_is_uniform_by_default():
    s = BayesianScorer(CONSTRAINTS)
    assert s.trust == pytest.approx(0.5)
    lo, hi = s.credible_interval()
    assert lo == pytest.approx(0.025) and hi == pytest.approx(0.975)


def test_clean_turns_raise_trust_and_narrow_the_interval():
    s, trace = BayesianScorer(CONSTRAINTS, discount=1.0), Trace()
    widths = []
    for _ in range(30):
        a = s.score_turn(turn_with(trace))
        widths.append(a.ci_high - a.ci_low)
    assert a.trust == pytest.approx(31 / 32)  # Beta(1 + 30, 1)
    assert widths == sorted(widths, reverse=True)
    assert a.risk is RiskLevel.NOMINAL
    assert s.alerts == []


def test_violation_triggers_high_risk_alert():
    received = []
    s, trace = BayesianScorer(CONSTRAINTS, on_alert=received.append), Trace()
    for _ in range(20):
        s.score_turn(turn_with(trace))
    before = s.trust

    a = s.score_turn(turn_with(trace, power=140.0))  # outside the actuator range

    assert a.risk is RiskLevel.HIGH
    assert a.trust < before
    assert [r.constraint for r in a.violations] == ["action.power in [0, 100]"]
    assert len(received) == 1 and received[0].turn == 20
    assert "power = 140" in received[0].message


def test_worse_violation_is_stronger_evidence():
    def trust_after(power):
        s, trace = BayesianScorer(CONSTRAINTS), Trace()
        for _ in range(10):
            s.score_turn(turn_with(trace))
        return s.score_turn(turn_with(trace, power=power)).trust

    assert trust_after(250.0) < trust_after(105.0)


def test_each_constraint_type_detects_its_violation():
    s, trace = BayesianScorer(CONSTRAINTS), Trace()
    s.score_turn(turn_with(trace))
    a = s.score_turn(turn_with(trace, expected=60.0, observed=75.0))
    assert {r.constraint for r in a.violations} == {
        "observed.temp in [20, 70]",
        "|d observed.temp| <= 3",
        "|expected - observed temp| <= 2",
    }


def test_sustained_violations_become_critical():
    s, trace = BayesianScorer(CONSTRAINTS), Trace()
    for _ in range(20):
        s.score_turn(turn_with(trace))
    levels = [s.score_turn(turn_with(trace, power=150.0)).risk for _ in range(15)]
    assert levels[0] is RiskLevel.HIGH
    assert levels[-1] is RiskLevel.CRITICAL
    assert s.trust < s.trust_threshold


def test_discount_lets_trust_recover_and_forget():
    s, trace = BayesianScorer(CONSTRAINTS, discount=0.8), Trace()
    for _ in range(5):
        s.score_turn(turn_with(trace, power=150.0))
    low = s.trust
    for _ in range(40):
        s.score_turn(turn_with(trace))
    assert s.trust > 0.8 > low


def test_turn_without_checkable_data_is_not_evidence():
    s, trace = BayesianScorer(CONSTRAINTS), Trace()
    with trace.turn():
        trace.think("just thinking")
    a = s.observe(trace)[0]
    assert (a.alpha, a.beta) == (1.0, 1.0) and a.results == []


def test_predicate_constraint():
    c = PredicateConstraint("has a thought", lambda t, prev: bool(t.thoughts))
    s, trace = BayesianScorer([c]), Trace()
    a = s.score_turn(turn_with(trace))  # turn_with records no thought
    assert [r.constraint for r in a.violations] == ["has a thought"]
    # a violation on the very first turn drags a uniform prior under the threshold
    assert a.risk is RiskLevel.CRITICAL


def test_observe_is_incremental_and_attach_scores_live():
    trace = Trace()
    s = BayesianScorer(CONSTRAINTS)
    turn_with(trace)
    turn_with(trace)
    assert len(s.observe(trace)) == 2
    assert s.observe(trace) == []
    turn_with(trace)
    assert [a.turn for a in s.observe(trace)] == [2]

    live = BayesianScorer(CONSTRAINTS)
    trace2 = Trace()
    live.attach(trace2)
    turn_with(trace2)
    assert len(live.history) == 1


def test_trust_decays_once_the_simulated_agent_drifts():
    sim = ThermalLoopSim(drift_onset=40, seed=3)
    trace, s = Trace(), BayesianScorer(sim.constraints())
    s.attach(trace)
    for _ in range(90):
        sim.step(trace)

    trust = [a.trust for a in s.history]
    assert not any(a.violations for a in s.history[:40])
    assert trust[39] > 0.9
    assert trust[-1] < 0.2
    assert s.history[-1].risk is RiskLevel.CRITICAL
    assert s.alerts and s.alerts[0].turn > 40
    assert s.records()[-1]["risk"] == "CRITICAL"


def test_healthy_agent_never_alerts():
    sim = ThermalLoopSim(drift_onset=None, seed=1)
    trace, s = Trace(), BayesianScorer(sim.constraints())
    s.attach(trace)
    for _ in range(200):
        sim.step(trace)
    assert s.alerts == []
    assert s.history[-1].risk is RiskLevel.NOMINAL
