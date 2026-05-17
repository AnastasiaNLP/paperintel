from models.agent_runs import AgentRun
from models.conversation_state import ConversationState, add_lists
from models.errors import make_error


def test_conversation_state_partial_fields_allowed():
    state: ConversationState = {
        "session_id": "session-1",
        "needs_discovery": True,
        "discovery_topic": "agent memory",
    }

    assert state["session_id"] == "session-1"
    assert state["needs_discovery"] is True
    assert state["discovery_topic"] == "agent memory"


def test_conversation_state_accumulates_agent_runs():
    first = AgentRun(agent_name="intent_router")
    second = AgentRun(agent_name="answer_agent")

    assert add_lists([first], [second]) == [first, second]


def test_conversation_state_accumulates_errors():
    first = make_error("WARNING", "first")
    second = make_error("WARNING", "second")

    assert add_lists([first], [second]) == [first, second]
