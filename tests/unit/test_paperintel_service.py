import pytest

from api.in_memory_session_store import SessionNotFoundError
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.discovery import SearchCandidate
from models.api import HealthStatus
from models.errors import ErrorCodes, make_error
from models.session import HandlerResult, Session, Turn
from services.paperintel_service import (
    ComparisonNotFoundError,
    InvalidSessionPhaseError,
    NoActivePapersError,
    PaperIntelService,
    PaperWorkspaceNotFoundError,
)
from services.selected_candidate_resolver import NoSelectedCandidatesError


class FakeHandler:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.created_sessions = []
        self.messages = []
        self.selected_analysis_calls = []
        self.selected_analysis_result = None

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

    def analyze_selected_papers(self, session_id, urls):
        self.selected_analysis_calls.append((session_id, list(urls)))
        return self.selected_analysis_result or HandlerResult(
            session_id=session_id,
            response_text="selected analysis complete",
            phase="qa",
            intent="analyze_paper",
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


class FakeSelectedCandidateResolver:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = []

    def resolve(self, session_id):
        self.calls.append(session_id)

        class Selected:
            def __init__(self, candidates):
                self.candidates = candidates

            @property
            def urls(self):
                return [candidate.url for candidate in self.candidates]

            @property
            def candidate_ids(self):
                return [candidate.id for candidate in self.candidates]

        return Selected(self.candidates)


class FailingSelectedCandidateResolver:
    def resolve(self, session_id):
        raise NoSelectedCandidatesError(session_id)


class FakeCandidateRepository:
    def __init__(self):
        self.updates = []

    def update_status(self, candidate_id, status):
        self.updates.append((candidate_id, status))
        return None


class FakeArtifactRepository:
    def __init__(self, *, session_id: str = "session-1") -> None:
        self.workspaces = [
            PaperWorkspace(
                session_id=session_id,
                paper_id="1706.03762",
                title="Attention Is All You Need",
                source_url="https://arxiv.org/abs/1706.03762",
                pipeline_stage="completed",
                full_markdown_report="# Report",
            )
        ]
        self.comparison = ComparisonArtifact(
            session_id=session_id,
            paper_ids=["1706.03762", "2401.00001"],
            comparison_markdown="# Comparison",
        )

    def list_workspaces(self, session_id):
        return [
            workspace
            for workspace in self.workspaces
            if workspace.session_id == session_id
        ]

    def get_workspace(self, session_id, paper_id):
        for workspace in self.workspaces:
            if workspace.session_id == session_id and workspace.paper_id == paper_id:
                return workspace
        return None

    def latest_comparison(self, session_id):
        if self.comparison.session_id == session_id:
            return self.comparison
        return None


def _candidate(candidate_id: str) -> SearchCandidate:
    return SearchCandidate(
        id=candidate_id,
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        status="selected",
        title=f"Paper {candidate_id}",
        url=f"https://arxiv.org/abs/{candidate_id}",
        arxiv_id=candidate_id,
    )


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


def test_service_synthesize_papers_uses_default_prompt():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    handler.store.sessions[session.id] = session.model_copy(
        update={"active_paper_ids": ["paper-1", "paper-2"]}
    )

    result = service.synthesize_papers(session.id)

    assert result.response_text.startswith("handled: Synthesize the active papers")
    assert handler.messages == [
        (
            session.id,
            (
                "Synthesize the active papers. Compare their main contributions, "
                "methods, trade-offs, limitations, and practical implications. "
                "Ground the answer in the papers and include citations."
            ),
        )
    ]


def test_service_synthesize_papers_uses_custom_prompt():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    handler.store.sessions[session.id] = session.model_copy(
        update={"active_paper_ids": ["paper-1"]}
    )

    result = service.synthesize_papers(session.id, "Compare deployment risks.")

    assert result.response_text == "handled: Compare deployment risks."
    assert handler.messages == [(session.id, "Compare deployment risks.")]


def test_service_synthesize_papers_requires_active_papers():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(NoActivePapersError):
        service.synthesize_papers(session.id)


def test_service_discover_papers_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.discover_papers(session.id, "Find papers about agent memory")

    assert result.response_text == "handled: Find papers about agent memory"
    assert handler.messages == [(session.id, "Find papers about agent memory")]


def test_service_discover_papers_wraps_bare_topic_for_routing():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.discover_papers(session.id, "agent memory")

    assert result.response_text == "handled: Find papers about agent memory"
    assert handler.messages == [(session.id, "Find papers about agent memory")]


def test_service_select_papers_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    handler.store.sessions[session.id] = session.model_copy(update={"phase": "selection"})

    result = service.select_papers(session.id, "use 1 and 3")

    assert result.response_text == "handled: use 1 and 3"
    assert handler.messages == [(session.id, "use 1 and 3")]


def test_service_select_papers_requires_selection_phase():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(InvalidSessionPhaseError):
        service.select_papers(session.id, "use 1")


def test_service_analyze_selected_papers_resolves_and_updates_statuses():
    handler = FakeHandler()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver(
            [_candidate("2401.1"), _candidate("2401.2")]
        ),
        candidate_repository=FakeCandidateRepository(),
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.response_text == "selected analysis complete"
    assert handler.selected_analysis_calls == [
        (
            session.id,
            ["https://arxiv.org/abs/2401.1", "https://arxiv.org/abs/2401.2"],
        )
    ]
    assert service.candidate_repository.updates == [
        ("2401.1", "analyzed"),
        ("2401.2", "analyzed"),
    ]


def test_service_analyze_selected_papers_updates_statuses_with_warnings():
    handler = FakeHandler()
    handler.selected_analysis_result = HandlerResult(
        session_id="session-1",
        response_text="selected analysis complete",
        phase="qa",
        intent="analyze_paper",
        errors=[
            make_error(
                ErrorCodes.WARNING,
                "benchmarks missing",
                severity="warning",
                recoverable=True,
            )
        ],
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )
    repository = FakeCandidateRepository()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver([_candidate("2401.1")]),
        candidate_repository=repository,
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.errors
    assert repository.updates == [("2401.1", "analyzed")]


def test_service_analyze_selected_papers_does_not_update_status_when_analysis_missing():
    handler = FakeHandler()
    handler.selected_analysis_result = HandlerResult(
        session_id="session-1",
        response_text="analysis missing",
        phase="selection",
        intent="analyze_paper",
        needs_analysis=True,
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )
    repository = FakeCandidateRepository()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver([_candidate("2401.1")]),
        candidate_repository=repository,
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.needs_analysis is True
    assert repository.updates == []


def test_service_analyze_selected_papers_requires_resolver():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(RuntimeError):
        service.analyze_selected_papers(session.id)


def test_service_analyze_selected_papers_propagates_resolver_error():
    service = PaperIntelService(
        handler=FakeHandler(),
        selected_candidate_resolver=FailingSelectedCandidateResolver(),
        candidate_repository=FakeCandidateRepository(),
    )
    session = service.create_session()

    with pytest.raises(NoSelectedCandidatesError):
        service.analyze_selected_papers(session.id)


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


def test_service_list_paper_workspaces_returns_repository_results():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    workspaces = service.list_paper_workspaces(session.id)

    assert [workspace.paper_id for workspace in workspaces] == ["1706.03762"]


def test_service_get_paper_workspace_returns_workspace():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    workspace = service.get_paper_workspace(session.id, "1706.03762")

    assert workspace.title == "Attention Is All You Need"


def test_service_get_paper_workspace_raises_404_domain_error():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    with pytest.raises(PaperWorkspaceNotFoundError):
        service.get_paper_workspace(session.id, "missing")


def test_service_get_latest_comparison_returns_artifact():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    comparison = service.get_latest_comparison(session.id)

    assert comparison.paper_ids == ["1706.03762", "2401.00001"]


def test_service_get_latest_comparison_raises_when_missing():
    handler = FakeHandler()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(),
    )
    session = service.create_session()

    with pytest.raises(ComparisonNotFoundError):
        service.get_latest_comparison(session.id)


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
