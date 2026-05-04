import pytest

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore, SessionNotFoundError
from models.errors import ErrorCodes, StructuredError


class FakeRunner:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {
            "response_text": "assistant response",
            "intent": "qa",
            "referenced_paper_ids": ["paper-1"],
            "artifact_refs": ["artifact-1"],
            "next_phase": "qa",
        }
        self.error = error
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        if self.error is not None:
            raise self.error
        return self.result


def _handler(runner=None, persistence=None):
    store = InMemorySessionStore()
    runner = runner or FakeRunner()
    persistence = persistence or InMemoryAgentRunPersistence()
    return (
        ChatHandler(
            store=store,
            conversation_runner=runner,
            agent_run_persistence=persistence,
        ),
        store,
        runner,
        persistence,
    )


def test_create_session_explicitly_sets_phase_and_persona():
    handler, store, _, _ = _handler()

    session = handler.create_session(
        persona="researcher",
        original_query="agent memory",
    )

    stored = store.require_session(session.id)
    assert stored.id == session.id
    assert stored.phase == "idle"
    assert stored.persona == "researcher"
    assert stored.original_query == "agent memory"


def test_handle_message_requires_existing_session():
    handler, _, _, _ = _handler()

    with pytest.raises(SessionNotFoundError):
        handler.handle_message("missing", "hello")


def test_handle_message_writes_user_turn_before_graph_and_assistant_after():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is in this paper?")

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "What is in this paper?"
    assert turns[1].content == "assistant response"
    assert turns[1].intent == "qa"
    assert turns[1].referenced_paper_ids == ["paper-1"]
    assert turns[1].artifact_refs == ["artifact-1"]
    assert result.user_turn_id == turns[0].id
    assert result.assistant_turn_id == turns[1].id
    assert runner.calls[0]["input"]["message"] == "What is in this paper?"


def test_handle_message_updates_phase_from_graph_result():
    handler, store, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "answer this")

    assert result.phase == "qa"
    assert store.require_session(session.id).phase == "qa"


def test_handler_propagates_session_id_and_persistence_to_graph_config():
    persistence = InMemoryAgentRunPersistence()
    handler, _, runner, _ = _handler(persistence=persistence)
    session = handler.create_session()

    handler.handle_message(session.id, "hello")

    call = runner.calls[0]
    assert call["input"]["session_id"] == session.id
    assert call["input"]["phase"] == "idle"
    assert call["input"]["persona"] == "engineer"
    assert call["config"]["configurable"]["session_id"] == session.id
    assert call["config"]["configurable"]["agent_run_persistence"] is persistence


def test_graph_failure_preserves_user_turn_and_writes_structured_error_turn():
    runner = FakeRunner(error=RuntimeError("boom"))
    handler, store, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "fail please")

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "fail please"
    assert isinstance(turns[1].error, StructuredError)
    assert turns[1].error.code == ErrorCodes.FATAL_ERROR
    assert turns[1].error.session_id == session.id
    assert turns[1].error.details["exception_type"] == "RuntimeError"
    assert result.phase == "failed"
    assert result.error == turns[1].error
    assert store.require_session(session.id).phase == "failed"


def test_two_messages_in_same_session_accumulate_turn_history():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    handler.handle_message(session.id, "first")
    handler.handle_message(session.id, "second")

    turns = store.list_recent_turns(session.id)
    assert [turn.content for turn in turns] == [
        "first",
        "assistant response",
        "second",
        "assistant response",
    ]
    assert len(runner.calls) == 2
    assert runner.calls[1]["input"]["phase"] == "qa"


def test_graph_invocation_result_and_handler_result_are_separate_types():
    handler, _, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "hello")

    assert hasattr(result, "user_turn_id")
    assert not hasattr(result, "raw")
