from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from agents.agent_run_recorder import AgentRunPersistence
from api.in_memory_session_store import SessionNotFoundError
from api.session_store import SessionStore
from models.agent_runs import AgentRun
from models.errors import StructuredError
from models.session import Persona, Session, SessionPhase, Turn, TurnRole
from storage.mappers import (
    agent_run_to_orm,
    orm_to_agent_run,
    orm_to_session,
    orm_to_structured_error,
    orm_to_turn,
    session_to_orm,
    structured_error_to_orm,
    turn_to_orm,
)
from storage.models import AgentRunORM, SessionORM, StructuredErrorORM, TurnORM


class PostgresSessionStore(SessionStore):
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        session = Session(persona=persona, original_query=original_query)
        with self.session_factory() as db:
            db.add(session_to_orm(session))
            db.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            return orm_to_session(orm) if orm is not None else None

    def require_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session

    def update_phase(self, session_id: str, phase: SessionPhase) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            orm.phase = phase
            db.commit()
            db.refresh(orm)
            return orm_to_session(orm)

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
        with self.session_factory() as db:
            if db.get(SessionORM, session_id) is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            error_id = None
            if isinstance(error, StructuredError):
                if error.session_id is None:
                    error = error.model_copy(update={"session_id": session_id})
                db.merge(structured_error_to_orm(error))
                error_id = error.id

            turn = Turn(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                referenced_paper_ids=referenced_paper_ids or [],
                artifact_refs=artifact_refs or [],
                error=error if isinstance(error, StructuredError) else None,
                metadata=metadata or {},
            )
            db.add(turn_to_orm(turn, error_id=error_id))
            db.commit()
            return turn

    def list_recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        with self.session_factory() as db:
            if db.get(SessionORM, session_id) is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            rows = (
                db.execute(
                    select(TurnORM)
                    .where(TurnORM.session_id == session_id)
                    .order_by(TurnORM.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [orm_to_turn(row) for row in reversed(rows)]


class PostgresAgentRunPersistence(AgentRunPersistence):
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def save(self, run: AgentRun) -> None:
        with self.session_factory() as db:
            db.merge(agent_run_to_orm(run))
            db.commit()

    def get(self, run_id: str) -> AgentRun | None:
        with self.session_factory() as db:
            orm = db.get(AgentRunORM, run_id)
            return orm_to_agent_run(orm) if orm is not None else None


class PostgresStructuredErrorRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def save(self, error: StructuredError) -> StructuredError:
        with self.session_factory() as db:
            db.merge(structured_error_to_orm(error))
            db.commit()
        return error

    def list_for_session(self, session_id: str) -> list[StructuredError]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(StructuredErrorORM)
                    .where(StructuredErrorORM.session_id == session_id)
                    .order_by(StructuredErrorORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_structured_error(row) for row in rows]


def clear_foundation_tables(db: DbSession) -> None:
    db.execute(delete(TurnORM))
    db.execute(delete(AgentRunORM))
    db.execute(delete(StructuredErrorORM))
    db.execute(delete(SessionORM))
    db.commit()
