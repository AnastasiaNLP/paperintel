import pytest

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore, SessionNotFoundError
from models.discovery import SearchCandidate
from models.errors import ErrorCodes, StructuredError
from models.qa import AnswerDraft
from services.selection_parser import SelectionHandler


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


class FakeRetrievalLayer:
    pass


class FakeCandidateRepository:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.updated = []

    def list_latest_for_session(self, session_id: str):
        return list(self.candidates)

    def update_status(self, candidate_id: str, status: str):
        self.updated.append((candidate_id, status))
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate.model_copy(update={"status": status})
        return None


def _candidate(rank: int) -> SearchCandidate:
    arxiv_id = f"2401.0000{rank}"
    return SearchCandidate(
        id=f"candidate-{rank}",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=rank,
        title=f"Paper {rank}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
        year=2024,
    )


def _handler(runner=None, persistence=None, analysis_runner=None, retrieval_layer=None):
    store = InMemorySessionStore()
    runner = runner or FakeRunner()
    persistence = persistence or InMemoryAgentRunPersistence()
    return (
        ChatHandler(
            store=store,
            conversation_runner=runner,
            analysis_runner=analysis_runner,
            agent_run_persistence=persistence,
            retrieval_layer=retrieval_layer,
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
    assert runner.calls[0]["input"]["user_message"] == "What is in this paper?"
    assert "message" not in runner.calls[0]["input"]


def test_handle_message_updates_phase_from_graph_result():
    handler, store, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "answer this")

    assert result.phase == "qa"
    assert store.require_session(session.id).phase == "qa"


def test_handler_propagates_session_id_and_persistence_to_graph_config():
    persistence = InMemoryAgentRunPersistence()
    retrieval_layer = FakeRetrievalLayer()
    handler, store, runner, _ = _handler(
        persistence=persistence,
        retrieval_layer=retrieval_layer,
    )
    session = handler.create_session()
    store.add_active_paper(session.id, "paper-1")

    handler.handle_message(session.id, "hello")

    call = runner.calls[0]
    assert call["input"]["session_id"] == session.id
    assert call["input"]["persona"] == "engineer"
    assert call["input"]["referenced_paper_ids"] == ["paper-1"]
    assert call["config"]["configurable"]["session_id"] == session.id
    assert call["config"]["configurable"]["session_store"] is store
    assert call["config"]["configurable"]["agent_run_persistence"] is persistence
    assert call["config"]["configurable"]["retrieval_layer"] is retrieval_layer


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
    assert result.errors == [turns[1].error]
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
    assert runner.calls[1]["input"]["user_message"] == "second"


def test_graph_invocation_result_and_handler_result_are_separate_types():
    handler, _, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "hello")

    assert hasattr(result, "user_turn_id")
    assert not hasattr(result, "raw")


def test_conversation_answer_draft_becomes_response_text():
    answer = AnswerDraft(
        question="What is the method?",
        answer_text="It uses retrieval.",
        persona="engineer",
        citations=[],
    )
    runner = FakeRunner(result={"answer_draft": answer, "intent": "qa_factual"})
    handler, store, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is the method?")

    assert result.response_text == "It uses retrieval."
    assert result.citations == []
    assert store.list_recent_turns(session.id)[1].content == "It uses retrieval."


def test_conversation_clarification_becomes_response_text():
    runner = FakeRunner(
        result={
            "intent": "clarification_needed",
            "needs_clarification": True,
            "clarification_question": "Which paper do you mean?",
        }
    )
    handler, _, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "What about the second paper?")

    assert result.response_text == "Which paper do you mean?"
    assert result.intent == "clarification_needed"


def test_analyze_paper_without_analysis_runner_returns_controlled_response():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2310.06825")

    assert runner.calls == []
    assert result.intent == "analyze_paper"
    assert result.needs_analysis is True
    assert "analysis is configured" in result.response_text
    assert [turn.role for turn in store.list_recent_turns(session.id)] == [
        "user",
        "assistant",
    ]


def test_conversation_discovery_signal_becomes_controlled_response():
    runner = FakeRunner(
        result={
            "intent": "discover",
            "needs_discovery": True,
            "discovery_topic": "long context memory for agents",
        }
    )
    handler, _, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "Can you help with this topic?")

    assert result.intent == "discover"
    assert result.needs_discovery is True
    assert result.discovery_topic == "long context memory for agents"
    assert "Discovery is not configured" in result.response_text


def test_handler_routes_selection_phase_to_selection_handler():
    store = InMemorySessionStore()
    runner = FakeRunner()
    repository = FakeCandidateRepository([_candidate(1), _candidate(2), _candidate(3)])
    selection_handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )
    handler = ChatHandler(
        store=store,
        conversation_runner=runner,
        selection_handler=selection_handler,
    )
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 1 and 3")

    assert runner.calls == []
    assert result.intent == "select_papers"
    assert result.phase == "idle"
    assert result.referenced_paper_ids == ["2401.00001", "2401.00003"]
    assert repository.updated == [
        ("candidate-1", "selected"),
        ("candidate-3", "selected"),
    ]
    assert store.require_session(session.id).selected_candidate_ids == [
        "candidate-1",
        "candidate-3",
    ]
    assert "Selected 2 papers" in result.response_text


def test_handler_keeps_selection_phase_on_invalid_selection():
    store = InMemorySessionStore()
    runner = FakeRunner()
    repository = FakeCandidateRepository([_candidate(1), _candidate(2)])
    selection_handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )
    handler = ChatHandler(
        store=store,
        conversation_runner=runner,
        selection_handler=selection_handler,
    )
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 9")

    assert runner.calls == []
    assert result.phase == "selection"
    assert repository.updated == []
    assert "Available numbers" in result.response_text


def test_handler_selection_phase_without_selection_handler_returns_controlled_response():
    handler, store, runner, _ = _handler()
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 1")

    assert runner.calls == []
    assert result.intent == "select_papers"
    assert result.phase == "selection"
    assert "not configured" in result.response_text
