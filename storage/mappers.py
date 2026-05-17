from models.agent_runs import AgentRun
from models.discovery import SearchCandidate
from models.errors import StructuredError
from models.retrieval import ChunkLocation, ChunkSource, EvidenceArtifact, PaperChunk
from models.session import Session, Turn
from storage.models import (
    AgentRunORM,
    PaperChunkORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
)


def session_to_orm(session: Session) -> SessionORM:
    return SessionORM(
        id=session.id,
        persona=session.persona,
        original_query=session.original_query,
        phase=session.phase,
        selected_candidate_ids=session.selected_candidate_ids,
        active_paper_ids=session.active_paper_ids,
        latest_comparison_id=session.latest_comparison_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def orm_to_session(orm: SessionORM) -> Session:
    return Session(
        id=orm.id,
        persona=orm.persona,
        original_query=orm.original_query,
        phase=orm.phase,
        selected_candidate_ids=list(orm.selected_candidate_ids or []),
        active_paper_ids=list(orm.active_paper_ids or []),
        latest_comparison_id=orm.latest_comparison_id,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def structured_error_to_orm(error: StructuredError) -> StructuredErrorORM:
    return StructuredErrorORM(
        id=error.id,
        session_id=error.session_id,
        paper_id=error.paper_id,
        agent_run_id=error.agent_run_id,
        code=error.code,
        message=error.message,
        node=error.node,
        agent=error.agent,
        severity=error.severity,
        recoverable=error.recoverable,
        details_json=error.details,
    )


def orm_to_structured_error(orm: StructuredErrorORM) -> StructuredError:
    return StructuredError(
        id=orm.id,
        code=orm.code,
        message=orm.message,
        node=orm.node,
        agent=orm.agent,
        severity=orm.severity,
        recoverable=orm.recoverable,
        paper_id=orm.paper_id,
        session_id=orm.session_id,
        agent_run_id=orm.agent_run_id,
        details=orm.details_json or {},
    )


def turn_to_orm(turn: Turn, *, error_id: str | None = None) -> TurnORM:
    return TurnORM(
        id=turn.id,
        session_id=turn.session_id,
        role=turn.role,
        content=turn.content,
        intent=turn.intent,
        referenced_paper_ids=turn.referenced_paper_ids,
        artifact_refs=turn.artifact_refs,
        error_id=error_id,
        metadata_json=turn.metadata,
        created_at=turn.created_at,
    )


def orm_to_turn(orm: TurnORM) -> Turn:
    return Turn(
        id=orm.id,
        session_id=orm.session_id,
        role=orm.role,
        content=orm.content,
        intent=orm.intent,
        referenced_paper_ids=list(orm.referenced_paper_ids or []),
        artifact_refs=list(orm.artifact_refs or []),
        error=orm_to_structured_error(orm.error) if orm.error else None,
        metadata=orm.metadata_json or {},
        created_at=orm.created_at,
    )


def agent_run_to_orm(run: AgentRun) -> AgentRunORM:
    return AgentRunORM(
        id=run.id,
        session_id=run.session_id,
        job_id=run.job_id,
        agent_name=run.agent_name,
        input_refs=run.input_refs,
        output_ref=run.output_ref,
        confidence=run.confidence,
        model=run.model,
        tool_calls=run.tool_calls,
        iteration_count=run.iteration_count,
        llm_call_count=run.llm_call_count,
        termination_reason=run.termination_reason,
        status=run.status,
        tokens_used=run.tokens_used,
        cost_usd=run.cost_usd,
        details_json=run.details,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def orm_to_agent_run(orm: AgentRunORM) -> AgentRun:
    return AgentRun(
        id=orm.id,
        session_id=orm.session_id,
        job_id=orm.job_id,
        agent_name=orm.agent_name,
        input_refs=list(orm.input_refs or []),
        output_ref=orm.output_ref,
        confidence=orm.confidence,
        model=orm.model,
        tool_calls=list(orm.tool_calls or []),
        iteration_count=orm.iteration_count,
        llm_call_count=orm.llm_call_count,
        termination_reason=orm.termination_reason,
        status=orm.status,
        tokens_used=orm.tokens_used,
        cost_usd=orm.cost_usd,
        details=orm.details_json or {},
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def paper_chunk_to_orm(chunk: PaperChunk) -> PaperChunkORM:
    return PaperChunkORM(
        id=chunk.id,
        paper_id=chunk.paper_id,
        session_id=chunk.source.session_id,
        paper_index=chunk.source.paper_index,
        chunk_index=chunk.chunk_index,
        chunk_type=chunk.chunk_type,
        text=chunk.text,
        source_json=chunk.source.model_dump(mode="json"),
        location_json=chunk.location.model_dump(mode="json"),
        artifact_refs_json=[
            artifact.model_dump(mode="json") for artifact in chunk.artifact_refs
        ],
        metadata_json=chunk.metadata,
        embedding_model=chunk.embedding_model,
        embedding_dimensions=chunk.embedding_dimensions,
        created_at=chunk.created_at,
    )


def orm_to_paper_chunk(orm: PaperChunkORM) -> PaperChunk:
    return PaperChunk(
        id=orm.id,
        paper_id=orm.paper_id,
        chunk_index=orm.chunk_index,
        text=orm.text,
        chunk_type=orm.chunk_type,
        source=ChunkSource(**(orm.source_json or {})),
        location=ChunkLocation(**(orm.location_json or {})),
        artifact_refs=[
            EvidenceArtifact(**artifact)
            for artifact in list(orm.artifact_refs_json or [])
        ],
        metadata=orm.metadata_json or {},
        embedding_model=orm.embedding_model,
        embedding_dimensions=orm.embedding_dimensions,
        created_at=orm.created_at,
    )


def search_candidate_to_orm(candidate: SearchCandidate) -> SearchCandidateORM:
    return SearchCandidateORM(
        id=candidate.id,
        session_id=candidate.session_id,
        discovery_turn_id=candidate.discovery_turn_id,
        display_rank=candidate.display_rank,
        status=candidate.status,
        title=candidate.title,
        url=candidate.url,
        source=candidate.source,
        authors=candidate.authors,
        year=candidate.year,
        arxiv_id=candidate.arxiv_id,
        abstract=candidate.abstract,
        published_at=candidate.published_at,
        score=candidate.score,
        reasons=candidate.reasons,
        metadata_json=candidate.metadata,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )


def orm_to_search_candidate(orm: SearchCandidateORM) -> SearchCandidate:
    return SearchCandidate(
        id=orm.id,
        session_id=orm.session_id,
        discovery_turn_id=orm.discovery_turn_id,
        display_rank=orm.display_rank,
        status=orm.status,
        title=orm.title,
        url=orm.url,
        source=orm.source,
        authors=list(orm.authors or []),
        year=orm.year,
        arxiv_id=orm.arxiv_id,
        abstract=orm.abstract,
        published_at=orm.published_at,
        score=orm.score,
        reasons=list(orm.reasons or []),
        metadata=orm.metadata_json or {},
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )
