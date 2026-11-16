"""Generic trace model for an AI agent acting in a closed loop.

A :class:`Trace` is an append-only log of four kinds of event:

* :class:`AgentInput`      - what the agent was given (prompt, sensor snapshot)
* :class:`Thought`         - one link of its chain of thought (CoT)
* :class:`ToolCall`        - an action it took, plus the outcome it *expected*
* :class:`PhysicalFeedback`- what the physical world actually reported back

Events are grouped into *turns*: one pass around the perceive -> think -> act
-> observe loop. Scoring works turn by turn (see ``scorer.py``).
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(kw_only=True)
class Event:
    """Base class for everything a trace records."""

    kind: ClassVar[str] = "event"

    turn: int = 0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **asdict(self)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Event":
        data = dict(data)
        kind = data.pop("kind")
        try:
            cls = _EVENT_TYPES[kind]
        except KeyError:
            raise ValueError(f"unknown event kind: {kind!r}") from None
        return cls(**data)


@dataclass(kw_only=True)
class AgentInput(Event):
    """Something handed to the agent: a user instruction, a sensor snapshot."""

    kind: ClassVar[str] = "input"
    content: str
    role: str = "user"


@dataclass(kw_only=True)
class Thought(Event):
    """One step of the agent's chain of thought."""

    kind: ClassVar[str] = "thought"
    content: str


@dataclass(kw_only=True)
class ToolCall(Event):
    """An action. ``expected`` is the physical outcome the agent predicted,
    which is what lets us compare its beliefs against reality."""

    kind: ClassVar[str] = "tool_call"
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class PhysicalFeedback(Event):
    """A measurement from the physical world (sensor, simulator, actuator)."""

    kind: ClassVar[str] = "feedback"
    observed: dict[str, Any] = field(default_factory=dict)
    source: str = "sensor"
    call_id: str | None = None  # event_id of the ToolCall this answers, if any


_EVENT_TYPES: dict[str, type[Event]] = {
    cls.kind: cls for cls in (AgentInput, Thought, ToolCall, PhysicalFeedback)
}


@dataclass
class Turn:
    """Read-only view of the events in one loop iteration."""

    index: int
    events: list[Event]

    def _of(self, cls: type) -> list:
        return [e for e in self.events if isinstance(e, cls)]

    @property
    def inputs(self) -> list[AgentInput]:
        return self._of(AgentInput)

    @property
    def thoughts(self) -> list[Thought]:
        return self._of(Thought)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self._of(ToolCall)

    @property
    def feedback(self) -> list[PhysicalFeedback]:
        return self._of(PhysicalFeedback)

    @property
    def action(self) -> dict[str, Any]:
        """Tool-call arguments merged across the turn (later calls win)."""
        return _merge(c.arguments for c in self.tool_calls)

    @property
    def expected(self) -> dict[str, Any]:
        """What the agent predicted would happen."""
        return _merge(c.expected for c in self.tool_calls)

    @property
    def observed(self) -> dict[str, Any]:
        """What the physical world reported."""
        return _merge(f.observed for f in self.feedback)

    def values(self, source: str) -> dict[str, Any]:
        if source not in ("action", "expected", "observed"):
            raise ValueError(f"source must be action/expected/observed, got {source!r}")
        return getattr(self, source)


def _merge(dicts) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


TurnListener = Callable[[Turn], None]


class Trace:
    """Append-only record of one agent run.

    Typical use::

        trace = Trace(agent_id="arm-controller")
        with trace.turn():
            trace.input("hold 60 C")
            trace.think("reading is 58.9, raise heater a little")
            call = trace.tool_call("set_heater", {"power": 74}, expected={"temp": 59.6})
            trace.feedback({"temp": 59.5}, call_id=call.event_id)
    """

    def __init__(
        self,
        agent_id: str = "agent",
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.trace_id = trace_id or _new_id()
        self.metadata = metadata or {}
        self.events: list[Event] = []
        self._turn = 0
        self._turn_listeners: list[TurnListener] = []

    # -- recording ---------------------------------------------------------

    def record(self, event: Event) -> Event:
        event.turn = self._turn
        self.events.append(event)
        return event

    def input(self, content: str, role: str = "user", **metadata: Any) -> AgentInput:
        return self.record(AgentInput(content=content, role=role, metadata=metadata))

    def think(self, content: str, **metadata: Any) -> Thought:
        return self.record(Thought(content=content, metadata=metadata))

    def tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        expected: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ToolCall:
        return self.record(
            ToolCall(
                name=name,
                arguments=arguments or {},
                expected=expected or {},
                metadata=metadata,
            )
        )

    def feedback(
        self,
        observed: dict[str, Any],
        source: str = "sensor",
        call_id: str | None = None,
        **metadata: Any,
    ) -> PhysicalFeedback:
        return self.record(
            PhysicalFeedback(
                observed=observed, source=source, call_id=call_id, metadata=metadata
            )
        )

    # -- turns -------------------------------------------------------------

    @property
    def current_turn(self) -> int:
        """Index of the turn that new events are being written into."""
        return self._turn

    def end_turn(self) -> Turn:
        """Close the current turn, notify listeners, and open the next one."""
        turn = self.get_turn(self._turn)
        self._turn += 1
        for listener in self._turn_listeners:
            listener(turn)
        return turn

    @contextmanager
    def turn(self) -> Iterator[int]:
        """Record one loop iteration; the turn closes when the block exits."""
        try:
            yield self._turn
        finally:
            self.end_turn()

    def on_turn_end(self, listener: TurnListener) -> None:
        """Call ``listener(turn)`` every time a turn closes (live scoring)."""
        self._turn_listeners.append(listener)

    def get_turn(self, index: int) -> Turn:
        return Turn(index, [e for e in self.events if e.turn == index])

    def turns(self, include_open: bool = False) -> list[Turn]:
        """Closed turns in order. The turn still being written is left out
        unless ``include_open`` is set."""
        grouped: dict[int, list[Event]] = {}
        for e in self.events:
            grouped.setdefault(e.turn, []).append(e)
        return [
            Turn(i, evs)
            for i, evs in sorted(grouped.items())
            if include_open or i < self._turn
        ]

    def __len__(self) -> int:
        return len(self.events)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "current_turn": self._turn,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        trace = cls(
            agent_id=data.get("agent_id", "agent"),
            trace_id=data.get("trace_id"),
            metadata=data.get("metadata"),
        )
        trace.events = [Event.from_dict(e) for e in data.get("events", [])]
        last = max((e.turn for e in trace.events), default=-1)
        trace._turn = data.get("current_turn", last + 1)
        return trace

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "Trace":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
