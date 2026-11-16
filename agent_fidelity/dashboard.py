"""Live monitoring dashboard.

Run with::

    streamlit run agent_fidelity/dashboard.py

It drives the thermal-chamber scenario from ``simulate.py`` one turn at a time,
scores every turn with :class:`BayesianScorer`, and plots the trust decay curve
as the agent drifts away from physical reality.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from agent_fidelity import BayesianScorer, RiskLevel, Trace
from agent_fidelity.simulate import ThermalLoopSim

MAX_TURNS = 400
MIN_X_SPAN = 60

# Chart colours per theme: series hues are categorical slots 1-2, "critical" is
# the reserved status red and is only ever used for violation markers.
PALETTE = {
    "light": {
        "series1": "#2a78d6", "series2": "#eb6834", "critical": "#d03b3b",
        "surface": "#fcfcfb", "text2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
    },
    "dark": {
        "series1": "#3987e5", "series2": "#d95926", "critical": "#d03b3b",
        "surface": "#1a1a19", "text2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
    },
}

RISK_LABEL = {
    RiskLevel.NOMINAL: "✓ Nominal",
    RiskLevel.ELEVATED: "△ Elevated",
    RiskLevel.HIGH: "⚠ High",
    RiskLevel.CRITICAL: "⛔ Critical",
}


def colors() -> dict[str, str]:
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    return PALETTE["dark" if mode == "dark" else "light"]


# -- state ------------------------------------------------------------------


def make_scorer(sim: ThermalLoopSim, cfg: dict) -> BayesianScorer:
    return BayesianScorer(
        sim.constraints(),
        discount=cfg["discount"],
        trust_threshold=cfg["threshold"],
        credible_level=cfg["level"],
    )


def reset(cfg: dict) -> None:
    ss = st.session_state
    ss.sim = ThermalLoopSim(
        drift_onset=cfg["onset"] if cfg["drifts"] else None,
        bias_growth=cfg["bias_growth"],
        seed=cfg["seed"],
    )
    ss.trace = Trace(agent_id="thermal-agent")
    ss.scorer = make_scorer(ss.sim, cfg)
    ss.toasted = None  # risk level of the last turn a toast decision was made for
    ss.running = False


def advance(n: int) -> None:
    ss = st.session_state
    for _ in range(n):
        if ss.sim.t >= MAX_TURNS:
            ss.running = False
            break
        ss.sim.step(ss.trace)
    ss.scorer.observe(ss.trace)


# -- charts -----------------------------------------------------------------


def _style(chart: alt.LayerChart, c: dict[str, str], height: int) -> alt.LayerChart:
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=c["grid"], domainColor=c["axis"], tickColor=c["axis"],
            labelColor=c["muted"], titleColor=c["text2"], titleFontWeight="normal",
        )
        .configure_legend(labelColor=c["text2"], title=None, orient="top")
    )


def _x(span: int) -> alt.X:
    return alt.X(
        "turn:Q", title="Turn",
        scale=alt.Scale(domain=[0, span], nice=False),
        axis=alt.Axis(grid=False, tickMinStep=1),
    )


def _hover_rule(df: pd.DataFrame, x: alt.X, tooltip: list, c: dict[str, str]) -> alt.Chart:
    nearest = alt.selection_point(
        fields=["turn"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    return (
        alt.Chart(df)
        .mark_rule(color=c["muted"])
        .encode(x=x, opacity=alt.condition(nearest, alt.value(0.7), alt.value(0)), tooltip=tooltip)
        .add_params(nearest)
    )


def trust_chart(df: pd.DataFrame, scorer: BayesianScorer, onset: int | None) -> alt.LayerChart:
    c = colors()
    span = max(MIN_X_SPAN, int(df["turn"].max()) + 5)
    x = _x(span)
    pct = round(scorer.credible_level * 100)
    y = alt.Y(
        "ci_low:Q",
        title="Trust — P(a turn is physically faithful)",
        scale=alt.Scale(domain=[0, 1]),
    )
    base = alt.Chart(df)
    band = base.mark_area(opacity=0.18, color=c["series1"]).encode(x=x, y=y, y2="ci_high:Q")
    line = base.mark_line(strokeWidth=2, color=c["series1"]).encode(x=x, y="trust:Q")
    marks = (
        base.transform_filter("datum.violations > 0")
        .mark_point(shape="diamond", size=80, filled=True, opacity=1,
                    color=c["critical"], stroke=c["surface"], strokeWidth=1.5)
        .encode(x=x, y="trust:Q")
    )

    thr = pd.DataFrame({"turn": [0], "y": [scorer.trust_threshold],
                        "label": [f"trust threshold {scorer.trust_threshold:.2f}"]})
    layers = [
        band,
        alt.Chart(thr).mark_rule(strokeDash=[4, 4], color=c["muted"]).encode(y="y:Q"),
        alt.Chart(thr).mark_text(align="left", dx=6, dy=-8, fontSize=11, color=c["text2"])
        .encode(x=x, y="y:Q", text="label:N"),
    ]
    if onset is not None and onset <= df["turn"].max():
        on = pd.DataFrame({"turn": [onset], "y": [0.0], "label": ["drift injected"]})
        layers += [
            alt.Chart(on).mark_rule(strokeDash=[2, 3], color=c["muted"]).encode(x=x),
            alt.Chart(on).mark_text(align="left", dx=6, dy=-8, fontSize=11, color=c["text2"])
            .encode(x=x, y="y:Q", text="label:N"),
        ]
    layers += [
        line,
        marks,
        _hover_rule(df, x, [
            alt.Tooltip("turn:Q", title="Turn"),
            alt.Tooltip("trust:Q", title="Trust", format=".3f"),
            alt.Tooltip("ci_low:Q", title=f"{pct}% CI low", format=".3f"),
            alt.Tooltip("ci_high:Q", title=f"{pct}% CI high", format=".3f"),
            alt.Tooltip("risk:N", title="Risk"),
            alt.Tooltip("violated:N", title="Violated"),
        ], c),
    ]
    return _style(alt.layer(*layers), c, 340)


def telemetry_chart(wide: pd.DataFrame, limit: float) -> alt.LayerChart:
    c = colors()
    names = ["Sensor reading", "Agent's prediction"]
    span = max(MIN_X_SPAN, int(wide["turn"].max()) + 5)
    x = _x(span)
    long = wide.melt("turn", ["observed", "expected"], "series", "temp")
    long["series"] = long["series"].map({"observed": names[0], "expected": names[1]})
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("temp:Q", title="Chamber temperature (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(
                domain=names, range=[c["series1"], c["series2"]])),
        )
    )
    lim = pd.DataFrame({"turn": [0], "y": [limit], "label": [f"safe limit {limit:g} °C"]})
    return _style(
        alt.layer(
            alt.Chart(lim).mark_rule(strokeDash=[4, 4], color=c["muted"]).encode(y="y:Q"),
            alt.Chart(lim).mark_text(align="left", dx=6, dy=-8, fontSize=11, color=c["text2"])
            .encode(x=x, y="y:Q", text="label:N"),
            lines,
            _hover_rule(wide, x, [
                alt.Tooltip("turn:Q", title="Turn"),
                alt.Tooltip("observed:Q", title=names[0], format=".2f"),
                alt.Tooltip("expected:Q", title=names[1], format=".2f"),
                alt.Tooltip("power:Q", title="Heater command (%)", format=".1f"),
            ], c),
        ),
        c, 280,
    )


# -- page -------------------------------------------------------------------


def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Scenario")
    drifts = sb.toggle("Agent drifts", value=True,
                       help="Off: the agent stays healthy for the whole run.")
    onset = sb.slider("Drift starts at turn", 0, 200, 40, disabled=not drifts)
    bias_growth = sb.slider("Belief error growth (°C / turn)", 0.1, 2.0, 0.6, 0.1,
                            disabled=not drifts)
    seed = sb.number_input("Random seed", 0, 9999, 0)

    sb.header("Bayesian scorer")
    discount = sb.slider("Discount (memory)", 0.80, 1.00, 0.95, 0.01,
                         help="How much past evidence is kept each turn. 1.00 never forgets.")
    threshold = sb.slider("Trust threshold", 0.30, 0.95, 0.60, 0.05)
    level = sb.select_slider("Credible level", [0.80, 0.90, 0.95, 0.99], 0.95)

    sb.header("Playback")
    interval = sb.slider("Seconds per turn", 0.2, 2.0, 0.5, 0.1)

    return dict(drifts=drifts, onset=onset, bias_growth=bias_growth, seed=int(seed),
                discount=discount, threshold=threshold, level=level, interval=interval)


def render() -> None:
    ss = st.session_state
    scorer: BayesianScorer = ss.scorer
    if not scorer.history:
        st.info("Press **Start** to run the closed loop, or **Step** to advance one turn.")
        return

    last = scorer.history[-1]
    if last.violations and last.risk != ss.toasted:
        st.toast(scorer.alerts[-1].message, icon="⚠️")
    ss.toasted = last.risk

    pct = round(scorer.credible_level * 100)
    k = st.columns(4)
    k[0].metric("Trust (posterior mean)", f"{last.trust:.2f}")
    k[1].metric(f"{pct}% credible interval", f"{last.ci_low:.2f} – {last.ci_high:.2f}")
    k[2].metric("Risk", RISK_LABEL[last.risk])
    k[3].metric("Alerts", f"{len(scorer.alerts)}", help=f"over {len(scorer.history)} turns")

    if last.violations:
        st.error(scorer.alerts[-1].message, icon="🚨")
    elif last.risk is RiskLevel.ELEVATED:
        st.warning(
            f"Turn {last.turn} was clean, but the lower credible bound "
            f"({last.ci_low:.2f}) is still under the trust threshold.",
            icon="⚠️",
        )

    df = pd.DataFrame(scorer.records())
    st.subheader("Trust decay curve")
    st.caption(
        f"Line: posterior mean. Band: {pct}% credible interval. "
        "Red diamonds: turns that broke a physical constraint."
    )
    st.altair_chart(trust_chart(df, scorer, ss.sim.drift_onset), width="stretch")

    turns = ss.trace.turns()
    wide = pd.DataFrame(
        {
            "turn": t.index,
            "observed": t.observed.get("temp"),
            "expected": t.expected.get("temp"),
            "power": t.action.get("power"),
        }
        for t in turns
    )
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Belief vs. physical feedback")
        st.altair_chart(telemetry_chart(wide, ss.sim.temp_limits[1]), width="stretch")
    with right:
        st.subheader("Risk alerts")
        if scorer.alerts:
            log = pd.DataFrame(
                {
                    "Turn": a.turn,
                    "Level": RISK_LABEL[a.level],
                    "Trust": round(a.trust, 2),
                    "What happened": "; ".join(v.message for v in a.violations),
                }
                for a in reversed(scorer.alerts)
            )
            st.dataframe(
                log, hide_index=True, height=300, width="stretch",
                column_config={
                    "Turn": st.column_config.NumberColumn(width=50),
                    "Level": st.column_config.TextColumn(width=90),
                    "Trust": st.column_config.NumberColumn(width=50, format="%.2f"),
                },
            )
        else:
            st.caption("No constraint violations so far.")

    with st.expander(f"Latest turn in the trace (turn {turns[-1].index})"):
        t = turns[-1]
        a, b = st.columns(2)
        with a:
            st.markdown("**Input**")
            for e in t.inputs:
                st.write(e.content)
            st.markdown("**Chain of thought**")
            for e in t.thoughts:
                st.write(f"- {e.content}")
        with b:
            st.markdown("**Tool call**")
            for e in t.tool_calls:
                st.json({"name": e.name, "arguments": e.arguments, "expected": e.expected})
            st.markdown("**Physical feedback**")
            for e in t.feedback:
                st.json({"source": e.source, "observed": e.observed})

    with st.expander("Constraints and data table"):
        checks: dict[str, list[int]] = {}
        for h in scorer.history:
            for r in h.results:
                row = checks.setdefault(r.constraint, [0, 0])
                row[0] += 1
                row[1] += not r.satisfied
        st.dataframe(
            pd.DataFrame(
                {"Constraint": name, "Turns checked": n, "Violations": bad}
                for name, (n, bad) in checks.items()
            ),
            hide_index=True, width="stretch",
        )
        st.dataframe(df.join(wide.set_index("turn"), on="turn"), hide_index=True,
                     height=240, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Agent_Fidelity", page_icon="🛰️", layout="wide")
    ss = st.session_state
    cfg = sidebar()

    scenario = (cfg["drifts"], cfg["onset"], cfg["bias_growth"], cfg["seed"])
    scoring = (cfg["discount"], cfg["threshold"], cfg["level"])
    if "sim" not in ss or ss.get("scenario") != scenario:
        reset(cfg)  # a different scenario is a different run
    elif ss.get("scoring") != scoring:
        # same run, new scorer settings: re-score the trace already captured
        ss.scorer = make_scorer(ss.sim, cfg)
        ss.scorer.observe(ss.trace)
        ss.toasted = ss.scorer.history[-1].risk if ss.scorer.history else None
    ss.scenario, ss.scoring = scenario, scoring

    st.title("Agent_Fidelity")
    st.caption(
        "Closed-loop monitor: an agent holds a thermal chamber at 60 °C. Every turn its "
        "commands and predictions are checked against physical constraints, and a Bayesian "
        "posterior tracks how far it can still be trusted."
    )

    b = st.columns([2, 2, 2, 2, 3], vertical_alignment="center")
    if b[0].button("⏸ Pause" if ss.running else "▶ Start", type="primary", width="stretch"):
        ss.running = not ss.running
        st.rerun()
    if b[1].button("Step", disabled=ss.running, width="stretch"):
        advance(1)
    if b[2].button("+25 turns", disabled=ss.running, width="stretch"):
        advance(25)
    if b[3].button("Reset", width="stretch"):
        reset(cfg)
        st.rerun()
    status = b[4].empty()

    @st.fragment(run_every=cfg["interval"] if ss.running else None)
    def live() -> None:
        if ss.running:
            advance(1)
            if not ss.running:  # hit MAX_TURNS: stop the timer
                st.rerun()
        done = ss.sim.t >= MAX_TURNS
        status.caption(
            f"Turn {ss.sim.t} of {MAX_TURNS}"
            + (" · run finished, press Reset" if done else " · running" if ss.running else " · paused")
        )
        render()

    live()


main()
