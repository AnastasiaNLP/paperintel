from copy import deepcopy

from models.session import Persona, Session, SessionPhase, Turn, TurnRole, utc_now


class SessionNotFoundError(KeyError):
    pass


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._turns: dict[str, list[Turn]] = {}

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        session = Session(persona=persona, original_query=original_query)
        self._sessions[session.id] = session
        self._turns[session.id] = []
        return deepcopy(session)

    def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        return deepcopy(session) if session is not None else None

    def require_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session

    def update_phase(self, session_id: str, phase: SessionPhase) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        session.phase = phase
        session.updated_at = utc_now()
        return deepcopy(session)

    def add_active_paper(self, session_id: str, paper_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        if paper_id in session.active_paper_ids:
            return deepcopy(session)

        updated = session.model_copy(
            update={
                "active_paper_ids": [*session.active_paper_ids, paper_id],
                "updated_at": utc_now(),
            }
        )
        self._sessions[session_id] = updated
        return deepcopy(updated)

    def set_selected_candidate_ids(
        self,
        session_id: str,
        candidate_ids: list[str],
    ) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        deduped = list(dict.fromkeys(candidate_ids))
        updated = session.model_copy(
            update={
                "selected_candidate_ids": deduped,
                "updated_at": utc_now(),
            }
        )
        self._sessions[session_id] = updated
        return deepcopy(updated)

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
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        turn = Turn(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            referenced_paper_ids=referenced_paper_ids or [],
            artifact_refs=artifact_refs or [],
            error=error,
            metadata=metadata or {},
        )
        self._turns[session_id].append(turn)
        self._sessions[session_id].updated_at = utc_now()
        return deepcopy(turn)

    def list_recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        turns = self._turns[session_id][-limit:]
        return deepcopy(turns)
