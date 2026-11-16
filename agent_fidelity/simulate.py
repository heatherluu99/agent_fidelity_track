"""A small closed-loop scenario for demos and tests.

An agent holds a thermal chamber at a setpoint by commanding heater power.
Each turn it reads the temperature, reasons, calls ``set_heater`` with a
prediction of the next reading, and the plant answers with what really
happened.

From ``drift_onset`` onward the agent's internal belief about the temperature
picks up a growing cold bias (it thinks the chamber is cooler than it is), and
it increasingly forgets to clamp its own command. The failures arrive in
stages: first its predictions stop matching the sensor, then it issues
out-of-range commands, and finally the chamber itself overheats.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .constraints import (
    ConsistencyConstraint,
    PhysicalConstraint,
    RangeConstraint,
    RateConstraint,
)
from .trace import Trace, Turn


@dataclass
class ThermalLoopSim:
    setpoint: float = 60.0
    ambient: float = 22.0
    start_temp: float = 58.0
    drift_onset: int | None = 40  # turn at which the agent starts to drift; None = never
    bias_growth: float = 0.6  # deg C of extra belief error per turn after onset
    noise: float = 0.25  # sensor/process noise, deg C (std dev)
    seed: int | None = 0

    # plant: T' = T + gain * power - loss * (T - ambient)
    gain: float = 0.08
    loss: float = 0.15
    kp: float = 4.0

    temp_limits: tuple[float, float] = (20.0, 70.0)
    max_temp_rate: float = 3.0
    prediction_tolerance: float = 2.0

    temp: float = field(init=False)
    t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.temp = self.start_temp
        self._rng = random.Random(self.seed)

    def constraints(self) -> list[PhysicalConstraint]:
        lo, hi = self.temp_limits
        return [
            RangeConstraint("power", 0.0, 100.0, source="action", unit="%",
                            name="Heater command within 0-100%"),
            RangeConstraint("temp", lo, hi, source="observed", unit=" °C",
                            name=f"Chamber temperature within {lo:g}-{hi:g} °C"),
            RateConstraint("temp", self.max_temp_rate, unit=" °C",
                           name=f"Temperature slew ≤ {self.max_temp_rate:g} °C/turn"),
            ConsistencyConstraint("temp", self.prediction_tolerance, unit=" °C",
                                  name=f"Prediction matches sensor within {self.prediction_tolerance:g} °C"),
        ]

    def step(self, trace: Trace) -> Turn:
        """Run one perceive -> think -> act -> observe turn into ``trace``."""
        drifting = self.drift_onset is not None and self.t >= self.drift_onset
        age = self.t - self.drift_onset if drifting else 0
        bias = age * self.bias_growth
        believed = self.temp - bias

        feedforward = self.loss * (self.setpoint - self.ambient) / self.gain
        command = feedforward + self.kp * (self.setpoint - believed)
        forgets_clamp = drifting and self._rng.random() < min(0.5, 0.03 * age)
        if not forgets_clamp:
            command = min(max(command, 0.0), 100.0)

        applied = min(max(command, 0.0), 100.0)  # the actuator saturates regardless
        expected = believed + self.gain * applied - self.loss * (believed - self.ambient)
        next_temp = (
            self.temp
            + self.gain * applied
            - self.loss * (self.temp - self.ambient)
            + self._rng.gauss(0.0, self.noise)
        )

        with trace.turn():
            trace.input(
                f"Hold the chamber at {self.setpoint:.1f} °C. Sensor reads {self.temp:.2f} °C.",
                role="sensor",
            )
            trace.think(
                f"I believe the chamber is at {believed:.2f} °C, "
                f"{self.setpoint - believed:+.2f} °C from setpoint."
            )
            trace.think(
                f"Setting the heater to {command:.1f}% should bring it to {expected:.2f} °C."
            )
            call = trace.tool_call(
                "set_heater",
                {"power": round(command, 2)},
                expected={"temp": round(expected, 3)},
            )
            trace.feedback(
                {"temp": round(next_temp, 3), "applied_power": round(applied, 2)},
                source="chamber",
                call_id=call.event_id,
            )

        self.temp = next_temp
        self.t += 1
        return trace.get_turn(trace.current_turn - 1)
