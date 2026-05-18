from typing import Sequence, get_args

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from agents.agent_run_recorder import AgentRunPersistence
from api.in_memory_session_store import SessionNotFoundError
from api.session_store import SessionStore
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.discovery import CandidateStatus, SearchCandidate
from models.errors import StructuredError
from models.retrieval import PaperChunk, UpsertChunksResult
from models.session import Persona, Session, SessionPhase, Turn, TurnRole
from storage.mappers import (
    agent_run_to_orm,
    comparison_artifact_to_orm,
    orm_to_agent_run,
    orm_to_comparison_artifact,
    orm_to_paper_chunk,
    orm_to_paper_workspace,
    orm_to_session,
    orm_to_search_candidate,
    orm_to_structured_error,
    orm_to_turn,
    paper_chunk_to_orm,
    paper_workspace_to_orm,
    search_candidate_to_orm,
    session_to_orm,
    structured_error_to_orm,
    turn_to_orm,
)
from storage.models import (
    AgentRunORM,
    ComparisonArtifactORM,
    PaperChunkORM,
    PaperWorkspaceORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
)


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

    def add_active_paper(self, session_id: str, paper_id: str) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            current_ids = list(orm.active_paper_ids or [])
            if paper_id not in current_ids:
                orm.active_paper_ids = [*current_ids, paper_id]
                db.commit()
                db.refresh(orm)
            return orm_to_session(orm)

    def set_selected_candidate_ids(
        self,
        session_id: str,
        candidate_ids: list[str],
    ) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            orm.selected_candidate_ids = list(dict.fromkeys(candidate_ids))
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


class PostgresPaperChunkRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_many(self, chunks: list[PaperChunk]) -> UpsertChunksResult:
        if not chunks:
            return UpsertChunksResult()

        chunk_ids = [chunk.id for chunk in chunks]
        with self.session_factory() as db:
            existing_ids = set(
                db.execute(
                    select(PaperChunkORM.id).where(PaperChunkORM.id.in_(chunk_ids))
                )
                .scalars()
                .all()
            )
            for chunk in chunks:
                db.merge(paper_chunk_to_orm(chunk))
            db.commit()

        updated = len(existing_ids)
        inserted = len(chunks) - updated
        return UpsertChunksResult(inserted=inserted, updated=updated, skipped=0)

    def list_for_paper(self, paper_id: str) -> list[PaperChunk]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperChunkORM)
                    .where(PaperChunkORM.paper_id == paper_id)
                    .order_by(PaperChunkORM.chunk_index.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_paper_chunk(row) for row in rows]

    def get_many_by_ids(self, chunk_ids: Sequence[str]) -> list[PaperChunk]:
        if not chunk_ids:
            return []

        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperChunkORM).where(PaperChunkORM.id.in_(list(chunk_ids)))
                )
                .scalars()
                .all()
            )

        chunks_by_id = {row.id: orm_to_paper_chunk(row) for row in rows}
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]


class PostgresSearchCandidateRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_many(self, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        if not candidates:
            return []

        with self.session_factory() as db:
            for candidate in candidates:
                db.merge(search_candidate_to_orm(candidate))
            db.commit()
        return candidates

    def list_for_discovery_turn(
        self,
        session_id: str,
        discovery_turn_id: str,
    ) -> list[SearchCandidate]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(SearchCandidateORM)
                    .where(SearchCandidateORM.session_id == session_id)
                    .where(SearchCandidateORM.discovery_turn_id == discovery_turn_id)
                    .order_by(SearchCandidateORM.display_rank.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_search_candidate(row) for row in rows]

    def list_latest_for_session(self, session_id: str) -> list[SearchCandidate]:
        with self.session_factory() as db:
            latest_turn_id = (
                db.execute(
                    select(SearchCandidateORM.discovery_turn_id)
                    .where(SearchCandidateORM.session_id == session_id)
                    .order_by(SearchCandidateORM.created_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if latest_turn_id is None:
                return []

            rows = (
                db.execute(
                    select(SearchCandidateORM)
                    .where(SearchCandidateORM.session_id == session_id)
                    .where(SearchCandidateORM.discovery_turn_id == latest_turn_id)
                    .order_by(SearchCandidateORM.display_rank.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_search_candidate(row) for row in rows]

    def get_many_by_ids(self, candidate_ids: Sequence[str]) -> list[SearchCandidate]:
        if not candidate_ids:
            return []

        requested_ids = list(dict.fromkeys(candidate_ids))
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(SearchCandidateORM).where(
                        SearchCandidateORM.id.in_(requested_ids)
                    )
                )
                .scalars()
                .all()
            )
            by_id = {row.id: orm_to_search_candidate(row) for row in rows}
            return [by_id[candidate_id] for candidate_id in requested_ids if candidate_id in by_id]

    def update_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
    ) -> SearchCandidate | None:
        if status not in get_args(CandidateStatus):
            raise ValueError(f"Invalid search candidate status: {status}")

        with self.session_factory() as db:
            orm = db.get(SearchCandidateORM, candidate_id)
            if orm is None:
                return None
            orm.status = status
            db.commit()
            db.refresh(orm)
            return orm_to_search_candidate(orm)


class PostgresPaperWorkspaceRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_workspace(self, workspace: PaperWorkspace) -> PaperWorkspace:
        with self.session_factory() as db:
            existing = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == workspace.session_id)
                    .where(PaperWorkspaceORM.paper_id == workspace.paper_id)
                )
                .scalars()
                .first()
            )
            if existing is None:
                db.add(paper_workspace_to_orm(workspace))
                db.commit()
                return workspace

            existing.title = workspace.title
            existing.source_url = workspace.source_url
            existing.pipeline_stage = workspace.pipeline_stage
            existing.finalized_report_json = workspace.finalized_report_json
            existing.method_extraction_json = workspace.method_extraction_json
            existing.benchmarks_json = workspace.benchmarks_json
            existing.readiness_json = workspace.readiness_json
            existing.full_markdown_report = workspace.full_markdown_report
            db.commit()
            db.refresh(existing)
            return orm_to_paper_workspace(existing)

    def list_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == session_id)
                    .order_by(PaperWorkspaceORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_paper_workspace(row) for row in rows]

    def get_workspace(
        self,
        session_id: str,
        paper_id: str,
    ) -> PaperWorkspace | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == session_id)
                    .where(PaperWorkspaceORM.paper_id == paper_id)
                )
                .scalars()
                .first()
            )
            return orm_to_paper_workspace(orm) if orm is not None else None

    def save_comparison(
        self,
        artifact: ComparisonArtifact,
    ) -> ComparisonArtifact:
        with self.session_factory() as db:
            db.merge(comparison_artifact_to_orm(artifact))
            db.commit()
        return artifact

    def latest_comparison(self, session_id: str) -> ComparisonArtifact | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(ComparisonArtifactORM)
                    .where(ComparisonArtifactORM.session_id == session_id)
                    .order_by(ComparisonArtifactORM.created_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            return orm_to_comparison_artifact(orm) if orm is not None else None


def clear_foundation_tables(db: DbSession) -> None:
    db.execute(delete(ComparisonArtifactORM))
    db.execute(delete(PaperWorkspaceORM))
    db.execute(delete(SearchCandidateORM))
    db.execute(delete(PaperChunkORM))
    db.execute(delete(TurnORM))
    db.execute(delete(AgentRunORM))
    db.execute(delete(StructuredErrorORM))
    db.execute(delete(SessionORM))
    db.commit()
