from typing import Protocol

from api.chat_handler import ChatHandler
from models.api import HealthStatus
from models.discovery import CandidateStatus, SearchCandidate
from models.session import HandlerResult, Persona, Session, Turn
from services.selected_candidate_resolver import SelectedCandidateResolver


class InvalidSessionPhaseError(ValueError):
    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            f"Session is not in {expected} phase; current phase is {actual}."
        )
        self.expected = expected
        self.actual = actual


class NoActivePapersError(ValueError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} has no active papers. Analyze papers before synthesis."
        )
        self.session_id = session_id


class SearchCandidateRepository(Protocol):
    def update_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
    ) -> SearchCandidate | None:
        ...


class PaperIntelService:
    """
    Product-facing application facade for PaperIntel.

    Transport adapters should depend on this service instead of touching
    ChatHandler, graphs, or storage directly.
    """

    def __init__(
        self,
        *,
        handler: ChatHandler,
        health_checker=None,
        selected_candidate_resolver: SelectedCandidateResolver | None = None,
        candidate_repository: SearchCandidateRepository | None = None,
    ) -> None:
        self.handler = handler
        self.health_checker = health_checker
        self.selected_candidate_resolver = selected_candidate_resolver
        self.candidate_repository = candidate_repository

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        return self.handler.create_session(
            persona=persona,
            original_query=original_query,
        )

    def handle_message(self, session_id: str, message: str) -> HandlerResult:
        return self.handler.handle_message(session_id, message)

    def analyze_paper(self, session_id: str, paper_url: str) -> HandlerResult:
        return self.handler.handle_message(session_id, paper_url)

    def ask_question(self, session_id: str, question: str) -> HandlerResult:
        return self.handler.handle_message(session_id, question)

    def synthesize_papers(
        self,
        session_id: str,
        prompt: str | None = None,
    ) -> HandlerResult:
        session = self.handler.store.require_session(session_id)
        if not session.active_paper_ids:
            raise NoActivePapersError(session_id)
        question = (prompt or _DEFAULT_SYNTHESIS_PROMPT).strip()
        if not question:
            question = _DEFAULT_SYNTHESIS_PROMPT
        return self.handler.handle_message(session_id, question)

    def discover_papers(self, session_id: str, topic_message: str) -> HandlerResult:
        topic_message = topic_message.strip()
        if not _looks_like_discovery_message(topic_message):
            topic_message = f"Find papers about {topic_message}"
        return self.handler.handle_message(session_id, topic_message)

    def select_papers(self, session_id: str, selection_message: str) -> HandlerResult:
        session = self.handler.store.require_session(session_id)
        if session.phase != "selection":
            raise InvalidSessionPhaseError(expected="selection", actual=session.phase)
        return self.handler.handle_message(session_id, selection_message)

    def analyze_selected_papers(self, session_id: str) -> HandlerResult:
        if self.selected_candidate_resolver is None:
            raise RuntimeError("Selected candidate analysis is not configured.")
        if self.candidate_repository is None:
            raise RuntimeError("Search candidate repository is not configured.")

        selected = self.selected_candidate_resolver.resolve(session_id)
        result = self.handler.analyze_selected_papers(session_id, selected.urls)
        if result.intent == "analyze_paper" and not result.needs_analysis and not result.errors:
            for candidate_id in selected.candidate_ids:
                self.candidate_repository.update_status(candidate_id, "analyzed")
        return result

    def get_session(self, session_id: str) -> Session:
        return self.handler.store.require_session(session_id)

    def list_turns(self, session_id: str, *, limit: int = 50) -> list[Turn]:
        self.handler.store.require_session(session_id)
        return self.handler.store.list_recent_turns(session_id, limit=limit)

    def health(self) -> HealthStatus:
        if self.health_checker is None:
            return HealthStatus(healthy=True, checks={"basic": "ok"})
        return self.health_checker.check()


def _looks_like_discovery_message(message: str) -> bool:
    normalized = message.casefold()
    discovery_words = ("find", "search", "discover", "recommend")
    target_words = ("paper", "papers", "literature", "research")
    return any(word in normalized for word in discovery_words) and any(
        word in normalized for word in target_words
    )


_DEFAULT_SYNTHESIS_PROMPT = (
    "Synthesize the active papers. Compare their main contributions, methods, "
    "trade-offs, limitations, and practical implications. Ground the answer in "
    "the papers and include citations."
)
