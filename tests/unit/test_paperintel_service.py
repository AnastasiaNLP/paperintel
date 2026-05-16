import pytest

from api.in_memory_session_store import SessionNotFoundError
from models.api import HealthStatus
from models.session import HandlerResult, Session, Turn
from services.paperintel_service import PaperIntelService


class FakeHandler:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.created_sessions = []
        self.messages = []

    def create_session(self, *, persona="engineer", original_query=None):
        session = self.store.create_session(
            persona=persona,
            original_query=original_query,
        )
        self.created_sessions.append(session)
        return session

    def handle_message(self, session_id, message):
        self.messages.append((session_id, message))
        return HandlerResult(
            session_id=session_id,
            response_text=f"handled: {message}",
            phase="qa",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )


class FakeStore:
    def __init__(self) -> None:
        self.sessions = {}
        self.turns = {}

    def create_session(self, *, persona="engineer", original_query=None):
        session = Session(persona=persona, original_query=original_query)
        self.sessions[session.id] = session
        self.turns[session.id] = []
        return session

    def require_session(self, session_id):
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return self.sessions[session_id]

    def list_recent_turns(self, session_id, limit=20):
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return self.turns[session_id][-limit:]


class FakeHealthChecker:
    def __init__(self, status=None) -> None:
        self.status = status or HealthStatus(
            healthy=True,
            checks={"postgres": "ok", "qdrant": "ok"},
        )
        self.calls = 0

    def check(self):
        self.calls += 1
        return self.status


def test_service_create_session_delegates_to_handler_with_persona():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)

    session = service.create_session(
        persona="researcher",
        original_query="agent memory",
    )

    assert session.persona == "researcher"
    assert session.original_query == "agent memory"
    assert handler.created_sessions == [session]


def test_service_handle_message_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.handle_message(session.id, "hello")

    assert result.response_text == "handled: hello"
    assert handler.messages == [(session.id, "hello")]


def test_service_analyze_paper_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.analyze_paper(session.id, "https://arxiv.org/abs/1706.03762")

    assert result.response_text == "handled: https://arxiv.org/abs/1706.03762"
    assert handler.messages == [(session.id, "https://arxiv.org/abs/1706.03762")]


def test_service_ask_question_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.ask_question(session.id, "What is the contribution?")

    assert result.response_text == "handled: What is the contribution?"
    assert handler.messages == [(session.id, "What is the contribution?")]


def test_service_get_session_returns_session_from_store():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    assert service.get_session(session.id) == session


def test_service_get_session_raises_for_missing_session():
    service = PaperIntelService(handler=FakeHandler())

    with pytest.raises(SessionNotFoundError):
        service.get_session("missing")


def test_service_list_turns_returns_history_from_store():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    first = Turn(session_id=session.id, role="user", content="first")
    second = Turn(session_id=session.id, role="assistant", content="second")
    handler.store.turns[session.id] = [first, second]

    assert service.list_turns(session.id, limit=1) == [second]


def test_service_list_turns_requires_session_before_listing():
    service = PaperIntelService(handler=FakeHandler())

    with pytest.raises(SessionNotFoundError):
        service.list_turns("missing")


def test_service_health_without_checker_returns_basic_ok():
    service = PaperIntelService(handler=FakeHandler())

    status = service.health()

    assert status.healthy is True
    assert status.checks == {"basic": "ok"}


def test_service_health_uses_checker_when_configured():
    checker = FakeHealthChecker()
    service = PaperIntelService(handler=FakeHandler(), health_checker=checker)

    status = service.health()

    assert status.healthy is True
    assert status.checks["postgres"] == "ok"
    assert checker.calls == 1
