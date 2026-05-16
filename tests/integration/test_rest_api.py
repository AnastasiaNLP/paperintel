import asyncio

import httpx

from api.in_memory_session_store import SessionNotFoundError
from api.rest.app import create_rest_app
from models.api import HealthStatus
from models.session import HandlerResult, Session, Turn


class FakeService:
    def __init__(self) -> None:
        self.sessions = {
            "session-1": Session(
                id="session-1",
                persona="engineer",
                phase="qa",
                active_paper_ids=["1706.03762"],
            )
        }
        self.turns = [
            Turn(
                id="turn-1",
                session_id="session-1",
                role="user",
                content="What is the contribution?",
                intent="qa_factual",
                referenced_paper_ids=["1706.03762"],
            )
        ]
        self.created_payloads = []
        self.analyze_calls = []
        self.ask_calls = []
        self.health_status = HealthStatus(healthy=True, checks={"basic": "ok"})

    def create_session(self, *, persona="engineer", original_query=None):
        self.created_payloads.append(
            {"persona": persona, "original_query": original_query}
        )
        session = Session(
            id="created-session",
            persona=persona,
            original_query=original_query,
        )
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id):
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"Session not found: {session_id}") from exc

    def list_turns(self, session_id, *, limit=50):
        self.get_session(session_id)
        return self.turns[:limit]

    def analyze_paper(self, session_id, paper_url):
        self.get_session(session_id)
        self.analyze_calls.append((session_id, paper_url))
        return _handler_result(session_id=session_id, response_text="Analyzed paper.")

    def ask_question(self, session_id, question):
        self.get_session(session_id)
        self.ask_calls.append((session_id, question))
        return _handler_result(
            session_id=session_id,
            response_text="The answer.",
            intent="qa_factual",
            referenced_paper_ids=["1706.03762"],
        )

    def health(self):
        return self.health_status


class ExplodingService(FakeService):
    def ask_question(self, session_id, question):
        raise RuntimeError("traceback details should not leak")


def _handler_result(
    *,
    session_id: str = "session-1",
    response_text: str = "OK",
    intent: str | None = None,
    referenced_paper_ids: list[str] | None = None,
) -> HandlerResult:
    return HandlerResult(
        session_id=session_id,
        response_text=response_text,
        phase="qa",
        intent=intent,
        referenced_paper_ids=referenced_paper_ids or [],
        user_turn_id="turn-user",
        assistant_turn_id="turn-assistant",
    )


def _request(service, method: str, path: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(
            app=create_rest_app(service=service or FakeService()),
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_create_session_returns_session():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions",
        json={"persona": "researcher", "original_query": "memory agents"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "created-session"
    assert response.json()["persona"] == "researcher"
    assert service.created_payloads == [
        {"persona": "researcher", "original_query": "memory agents"}
    ]


def test_create_session_validates_persona():
    response = _request(None, "POST", "/sessions", json={"persona": "manager"})

    assert response.status_code == 422


def test_get_session_returns_session():
    response = _request(None, "GET", "/sessions/session-1")

    assert response.status_code == 200
    assert response.json()["active_paper_ids"] == ["1706.03762"]


def test_get_session_returns_404_for_missing_session():
    response = _request(None, "GET", "/sessions/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found"


def test_list_turns_returns_turns():
    response = _request(None, "GET", "/sessions/session-1/turns?limit=1")

    assert response.status_code == 200
    assert response.json()["turns"][0]["content"] == "What is the contribution?"
    assert response.json()["turns"][0]["referenced_paper_ids"] == ["1706.03762"]


def test_health_returns_service_health():
    response = _request(None, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "checks": {"basic": "ok"}}


def test_health_returns_503_when_unhealthy():
    service = FakeService()
    service.health_status = HealthStatus(
        healthy=False,
        checks={"postgres": "error:RuntimeError"},
    )

    response = _request(service, "GET", "/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_analyze_requires_valid_url():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/analyze",
        json={"paper_url": "arxiv 1706.03762"},
    )

    assert response.status_code == 422


def test_analyze_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/analyze",
        json={"paper_url": "https://arxiv.org/abs/1706.03762"},
    )

    assert response.status_code == 200
    assert response.json()["response_text"] == "Analyzed paper."
    assert service.analyze_calls[0][0] == "session-1"
    assert service.analyze_calls[0][1].startswith("https://arxiv.org/abs/1706.03762")


def test_ask_requires_non_empty_question():
    response = _request(None, "POST", "/sessions/session-1/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/ask",
        json={"question": "What is the contribution?"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "qa_factual"
    assert response.json()["referenced_paper_ids"] == ["1706.03762"]
    assert service.ask_calls == [("session-1", "What is the contribution?")]


def test_message_response_shape_excludes_internal_fields():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/ask",
        json={"question": "What is the contribution?"},
    )

    payload = response.json()
    assert "agent_runs" not in payload
    assert "errors" not in payload
    assert "user_turn_id" not in payload
    assert "assistant_turn_id" not in payload


def test_internal_error_returns_safe_500_without_traceback_leak():
    response = _request(
        ExplodingService(),
        "POST",
        "/sessions/session-1/ask",
        json={"question": "Boom?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "detail": "An internal error occurred while processing the request.",
    }
    assert "traceback" not in response.text.lower()
    assert "should not leak" not in response.text
