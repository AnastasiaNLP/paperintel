from api.chat_handler import ChatHandler
from models.api import HealthStatus
from models.session import HandlerResult, Persona, Session, Turn


class PaperIntelService:
    """
    Product-facing application facade for PaperIntel.

    Transport adapters should depend on this service instead of touching
    ChatHandler, graphs, or storage directly.
    """

    def __init__(self, *, handler: ChatHandler, health_checker=None) -> None:
        self.handler = handler
        self.health_checker = health_checker

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

    def get_session(self, session_id: str) -> Session:
        return self.handler.store.require_session(session_id)

    def list_turns(self, session_id: str, *, limit: int = 50) -> list[Turn]:
        self.handler.store.require_session(session_id)
        return self.handler.store.list_recent_turns(session_id, limit=limit)

    def health(self) -> HealthStatus:
        if self.health_checker is None:
            return HealthStatus(healthy=True, checks={"basic": "ok"})
        return self.health_checker.check()
