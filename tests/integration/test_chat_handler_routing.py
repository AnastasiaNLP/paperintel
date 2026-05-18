from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore
from models.agent_runs import AgentRun
from models.errors import ErrorCodes, StructuredError, make_error
from models.qa import AnswerDraft


class FakeRunner:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {}
        self.error = error
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        if self.error is not None:
            raise self.error
        return self.result


class FakeRetrievalLayer:
    pass


def _handler(
    *,
    conversation_result=None,
    analysis_result=None,
    conversation_error: Exception | None = None,
    retrieval_layer=None,
):
    store = InMemorySessionStore()
    conversation_runner = FakeRunner(conversation_result, conversation_error)
    analysis_runner = FakeRunner(analysis_result)
    persistence = InMemoryAgentRunPersistence()
    handler = ChatHandler(
        store=store,
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        agent_run_persistence=persistence,
        retrieval_layer=retrieval_layer,
    )
    return handler, store, conversation_runner, analysis_runner, persistence


def test_handler_routes_url_to_analysis_runner():
    handler, store, conversation_runner, analysis_runner, _ = _handler(
        analysis_result={
            "full_markdown_report": "# Analysis complete",
            "next_phase": "qa",
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2310.06825")

    assert conversation_runner.calls == []
    assert len(analysis_runner.calls) == 1
    assert analysis_runner.calls[0]["input"]["input_value"] == (
        "https://arxiv.org/abs/2310.06825"
    )
    assert result.intent == "analyze_paper"
    assert result.response_text == "# Analysis complete"
    assert result.phase == "qa"
    assert [turn.role for turn in store.list_recent_turns(session.id)] == [
        "user",
        "assistant",
    ]


def test_handler_preserves_comparison_markdown_from_analysis_result():
    handler, _, _, _, _ = _handler(
        analysis_result={
            "full_markdown_report": "# Analysis complete",
            "comparison_markdown": "# Paper Comparison\n\nA beats B on throughput.",
            "next_phase": "qa",
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2310.06825")

    assert result.intent == "analyze_paper"
    assert result.response_text == "# Analysis complete"
    assert result.comparison_markdown == "# Paper Comparison\n\nA beats B on throughput."


def test_handler_failed_analysis_result_sets_failed_phase():
    error = make_error(
        ErrorCodes.PAPER_ERROR,
        "arXiv metadata failed for 2605.13898",
        node="ingestion",
        severity="error",
        recoverable=False,
    )
    handler, store, _, analysis_runner, _ = _handler(
        analysis_result={
            "processing_stage": "failed",
            "paper_failed": True,
            "paper_failure_reason": "arXiv metadata failed for 2605.13898",
            "errors": [error],
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2605.13898")

    assert len(analysis_runner.calls) == 1
    assert result.intent == "analyze_paper"
    assert result.phase == "failed"
    assert result.response_text == "arXiv metadata failed for 2605.13898"
    assert result.errors == [error]
    assert store.require_session(session.id).phase == "failed"


def test_handler_routes_question_to_conversation_runner():
    answer = AnswerDraft(
        question="What is the method?",
        answer_text="It uses retrieval.",
        persona="engineer",
    )
    handler, store, conversation_runner, analysis_runner, _ = _handler(
        conversation_result={"answer_draft": answer, "intent": "qa_factual"}
    )
    session = handler.create_session()
    store.add_active_paper(session.id, "2310.06825")

    result = handler.handle_message(session.id, "What is the method?")

    assert len(conversation_runner.calls) == 1
    assert analysis_runner.calls == []
    assert conversation_runner.calls[0]["input"]["user_message"] == "What is the method?"
    assert conversation_runner.calls[0]["input"]["referenced_paper_ids"] == [
        "2310.06825"
    ]
    assert result.response_text == "It uses retrieval."
    assert result.intent == "qa_factual"


def test_handler_conversation_clarification_becomes_response_text():
    handler, _, conversation_runner, analysis_runner, _ = _handler(
        conversation_result={
            "intent": "clarification_needed",
            "needs_clarification": True,
            "clarification_question": "Which paper should I use?",
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "What about the second one?")

    assert len(conversation_runner.calls) == 1
    assert analysis_runner.calls == []
    assert result.response_text == "Which paper should I use?"


def test_handler_analyze_intent_without_url_returns_needs_analysis_response():
    handler, _, conversation_runner, analysis_runner, _ = _handler(
        conversation_result={
            "intent": "analyze_paper",
            "needs_analysis": True,
            "clarification_question": "Please send the paper URL directly.",
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "Analyze the paper I mentioned.")

    assert len(conversation_runner.calls) == 1
    assert analysis_runner.calls == []
    assert result.intent == "analyze_paper"
    assert result.needs_analysis is True
    assert result.response_text == "Please send the paper URL directly."


def test_handler_config_includes_all_conversation_dependencies():
    retrieval_layer = FakeRetrievalLayer()
    handler, store, conversation_runner, _, persistence = _handler(
        conversation_result={"response_text": "ok"},
        retrieval_layer=retrieval_layer,
    )
    session = handler.create_session()

    handler.handle_message(session.id, "Question")

    config = conversation_runner.calls[0]["config"]["configurable"]
    assert config["session_id"] == session.id
    assert config["session_store"] is store
    assert config["agent_run_persistence"] is persistence
    assert config["retrieval_layer"] is retrieval_layer


def test_handler_returns_safe_response_when_conversation_graph_crashes():
    handler, store, _, _, _ = _handler(
        conversation_error=RuntimeError("graph crash")
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is X?")

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "What is X?"
    assert "could not complete" in result.response_text.lower()
    assert result.phase == "failed"
    assert result.errors
    assert result.error == result.errors[0]


def test_handler_conversation_errors_return_safe_response():
    error = make_error(
        ErrorCodes.FATAL_ERROR,
        "planner failed",
        node="retrieval_planner",
        severity="error",
        recoverable=True,
        session_id="session-1",
    )
    handler, _, _, _, _ = _handler(
        conversation_result={
            "intent": "qa_factual",
            "errors": [error],
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is the method?")

    assert "could not complete" in result.response_text.lower()
    assert result.errors == [error]
    assert result.error == error


def test_handler_preserves_agent_runs_from_conversation_result():
    run = AgentRun(agent_name="answer_agent", session_id="session-1").complete()
    handler, _, _, _, _ = _handler(
        conversation_result={
            "response_text": "done",
            "agent_runs": [run],
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "Question")

    assert result.agent_runs == [run]
