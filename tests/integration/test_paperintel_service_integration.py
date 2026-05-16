from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.app_factory import create_paperintel_service
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore
from models.qa import AnswerDraft
from services.paperintel_service import PaperIntelService


class FakeRunner:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        return self.result


class FakeRetrievalLayer:
    pass


def _service(*, conversation_result=None, analysis_result=None):
    conversation_runner = FakeRunner(
        conversation_result
        or {
            "answer_draft": AnswerDraft(
                question="What is the method?",
                answer_text="It uses retrieval.",
                persona="engineer",
            ),
            "intent": "qa_factual",
            "referenced_paper_ids": ["1706.03762"],
        }
    )
    analysis_runner = FakeRunner(
        analysis_result
        or {
            "full_markdown_report": "# Analysis complete",
            "next_phase": "qa",
        }
    )
    handler = ChatHandler(
        store=InMemorySessionStore(),
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        agent_run_persistence=InMemoryAgentRunPersistence(),
    )
    return PaperIntelService(handler=handler), conversation_runner, analysis_runner


def test_service_full_session_lifecycle_with_chat_handler():
    service, conversation_runner, analysis_runner = _service()

    session = service.create_session(persona="engineer")
    analysis_result = service.analyze_paper(
        session.id,
        "https://arxiv.org/abs/1706.03762",
    )
    question_result = service.ask_question(session.id, "What is the method?")
    turns = service.list_turns(session.id)
    loaded = service.get_session(session.id)

    assert loaded.id == session.id
    assert analysis_result.intent == "analyze_paper"
    assert analysis_result.response_text == "# Analysis complete"
    assert question_result.intent == "qa_factual"
    assert question_result.response_text == "It uses retrieval."
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert len(analysis_runner.calls) == 1
    assert len(conversation_runner.calls) == 1


def test_service_health_uses_basic_ok_without_checker():
    service, _, _ = _service()

    assert service.health().healthy is True
    assert service.health().checks == {"basic": "ok"}


def test_app_factory_creates_paperintel_service_with_injected_dependencies():
    conversation_runner = FakeRunner({"response_text": "conversation"})
    analysis_runner = FakeRunner({"response_text": "analysis"})
    retrieval_layer = FakeRetrievalLayer()

    service = create_paperintel_service(
        database_url="sqlite:///:memory:",
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        retrieval_layer=retrieval_layer,
        enable_health_checks=False,
    )

    assert service.handler.conversation_runner is conversation_runner
    assert service.handler.analysis_runner is analysis_runner
    assert service.handler.retrieval_layer is retrieval_layer
    assert service.health().checks == {"basic": "ok"}
