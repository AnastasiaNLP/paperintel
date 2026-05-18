import asyncio

import httpx

from api.in_memory_session_store import SessionNotFoundError
from api.rest.app import create_rest_app
from models.api import HealthStatus
from models.discovery import SearchCandidate
from models.session import HandlerResult, Session, Turn
from services.paperintel_service import InvalidSessionPhaseError
from services.selected_candidate_resolver import (
    NoSelectedCandidatesError,
    SelectedCandidateNotReadyError,
)


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
        self.discover_calls = []
        self.select_calls = []
        self.analyze_selected_calls = []
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

    def discover_papers(self, session_id, topic):
        self.get_session(session_id)
        self.discover_calls.append((session_id, topic))
        return _handler_result(
            session_id=session_id,
            response_text="Here are candidate papers. Reply with numbers.",
            phase="selection",
            intent="discover",
            discovery_topic="agent memory",
            discovery_candidate_count=3,
        )

    def select_papers(self, session_id, selection):
        self.get_session(session_id)
        self.select_calls.append((session_id, selection))
        return _handler_result(
            session_id=session_id,
            response_text="Selected papers 1 and 3.",
            phase="idle",
            intent="select_papers",
            referenced_paper_ids=["2605.1", "2605.3"],
            selected_candidate_ids=["candidate-1", "candidate-3"],
        )

    def analyze_selected_papers(self, session_id):
        self.get_session(session_id)
        self.analyze_selected_calls.append(session_id)
        return _handler_result(
            session_id=session_id,
            response_text="Selected papers analyzed.",
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=["2605.1", "2605.3"],
            comparison_markdown="# Paper Comparison\n\n2605.1 vs 2605.3",
        )

    def health(self):
        return self.health_status


class ExplodingService(FakeService):
    def ask_question(self, session_id, question):
        raise RuntimeError("traceback details should not leak")


class WrongPhaseService(FakeService):
    def select_papers(self, session_id, selection):
        raise InvalidSessionPhaseError(expected="selection", actual="idle")


class NoSelectionService(FakeService):
    def analyze_selected_papers(self, session_id):
        raise NoSelectedCandidatesError(session_id)


class CandidateNotReadyService(FakeService):
    def analyze_selected_papers(self, session_id):
        raise SelectedCandidateNotReadyError(
            SearchCandidate(
                id="candidate-1",
                session_id=session_id,
                discovery_turn_id="turn-1",
                display_rank=1,
                status="proposed",
                title="Paper",
                url="https://arxiv.org/abs/2605.1",
            )
        )


def _handler_result(
    *,
    session_id: str = "session-1",
    response_text: str = "OK",
    phase: str = "qa",
    intent: str | None = None,
    referenced_paper_ids: list[str] | None = None,
    discovery_topic: str | None = None,
    discovery_candidate_count: int | None = None,
    selected_candidate_ids: list[str] | None = None,
    comparison_markdown: str | None = None,
) -> HandlerResult:
    return HandlerResult(
        session_id=session_id,
        response_text=response_text,
        phase=phase,
        intent=intent,
        referenced_paper_ids=referenced_paper_ids or [],
        discovery_topic=discovery_topic,
        discovery_candidate_count=discovery_candidate_count,
        selected_candidate_ids=selected_candidate_ids or [],
        comparison_markdown=comparison_markdown,
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


def test_discover_requires_non_empty_topic():
    response = _request(None, "POST", "/sessions/session-1/discover", json={"topic": ""})

    assert response.status_code == 422


def test_discover_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/discover",
        json={"topic": "Find papers about agent memory"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "discover"
    assert payload["phase"] == "selection"
    assert payload["discovery_topic"] == "agent memory"
    assert payload["discovery_candidate_count"] == 3
    assert service.discover_calls == [
        ("session-1", "Find papers about agent memory")
    ]


def test_discover_accepts_bare_topic():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/discover",
        json={"topic": "agent memory"},
    )

    assert response.status_code == 200
    assert service.discover_calls == [("session-1", "agent memory")]


def test_select_requires_non_empty_selection():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/select",
        json={"selection": ""},
    )

    assert response.status_code == 422


def test_select_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/select",
        json={"selection": "use 1 and 3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "select_papers"
    assert payload["phase"] == "idle"
    assert payload["referenced_paper_ids"] == ["2605.1", "2605.3"]
    assert payload["selected_candidate_ids"] == ["candidate-1", "candidate-3"]
    assert service.select_calls == [("session-1", "use 1 and 3")]


def test_select_returns_409_when_session_not_in_selection_phase():
    response = _request(
        WrongPhaseService(),
        "POST",
        "/sessions/session-1/select",
        json={"selection": "use 1"},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "invalid_session_phase"


def test_analyze_selected_calls_service_without_body():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "analyze_paper"
    assert payload["phase"] == "qa"
    assert payload["referenced_paper_ids"] == ["2605.1", "2605.3"]
    assert payload["comparison_markdown"] == "# Paper Comparison\n\n2605.1 vs 2605.3"
    assert service.analyze_selected_calls == ["session-1"]


def test_analyze_selected_returns_400_when_no_candidates_selected():
    response = _request(
        NoSelectionService(),
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "no_selected_candidates"


def test_analyze_selected_returns_409_when_candidate_not_ready():
    response = _request(
        CandidateNotReadyService(),
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 409
    assert response.json()["error"] == "selected_candidate_not_ready"


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
