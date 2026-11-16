from agent_fidelity import AgentInput, PhysicalFeedback, Thought, ToolCall, Trace


def _one_turn(trace: Trace, temp: float = 59.5) -> None:
    with trace.turn():
        trace.input("hold 60 C")
        trace.think("slightly cold, add power")
        call = trace.tool_call("set_heater", {"power": 74}, expected={"temp": 59.6})
        trace.feedback({"temp": temp}, call_id=call.event_id)


def test_turn_groups_all_four_event_kinds():
    trace = Trace(agent_id="a1")
    _one_turn(trace)
    _one_turn(trace, temp=59.9)

    turns = trace.turns()
    assert [t.index for t in turns] == [0, 1]
    t0 = turns[0]
    assert [type(e) for e in t0.events] == [AgentInput, Thought, ToolCall, PhysicalFeedback]
    assert t0.action == {"power": 74}
    assert t0.expected == {"temp": 59.6}
    assert t0.observed == {"temp": 59.5}
    assert t0.feedback[0].call_id == t0.tool_calls[0].event_id
    assert turns[1].observed == {"temp": 59.9}


def test_open_turn_is_hidden_until_closed():
    trace = Trace()
    trace.input("hello")
    assert trace.turns() == []
    assert len(trace.turns(include_open=True)) == 1
    trace.end_turn()
    assert len(trace.turns()) == 1


def test_turn_listener_fires_on_close():
    trace = Trace()
    seen = []
    trace.on_turn_end(lambda turn: seen.append(turn.index))
    _one_turn(trace)
    _one_turn(trace)
    assert seen == [0, 1]


def test_json_round_trip(tmp_path):
    trace = Trace(agent_id="a1", metadata={"run": 7})
    _one_turn(trace)
    trace.think("still mid-turn")

    path = tmp_path / "trace.json"
    trace.save(path)
    loaded = Trace.load(path)

    assert loaded.to_dict() == trace.to_dict()
    assert loaded.current_turn == 1
    assert isinstance(loaded.events[2], ToolCall)
    # recording continues in the right turn after a reload
    assert loaded.think("resumed").turn == 1
