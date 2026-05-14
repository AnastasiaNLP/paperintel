from typing import Protocol

from models.session import Persona, Session, SessionPhase, Turn, TurnRole


class SessionStore(Protocol):
    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        ...

    def get_session(self, session_id: str) -> Session | None:
        ...

    def require_session(self, session_id: str) -> Session:
        ...

    def update_phase(self, session_id: str, phase: SessionPhase) -> Session:
        ...

    def add_active_paper(self, session_id: str, paper_id: str) -> Session:
        ...

    def append_turn(
        self,
        session_id: str,
        *,
        role: TurnRole,
        content: str,
        intent: str | None = None,
        referenced_paper_ids: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        error=None,
        metadata: dict | None = None,
    ) -> Turn:
        ...

    def list_recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        ...
