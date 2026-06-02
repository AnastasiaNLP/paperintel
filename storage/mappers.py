from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_artifacts import BlobArtifact, BlobReference
from models.discovery import SearchCandidate
from models.external_metadata import ArxivMetadataCacheEntry
from models.errors import StructuredError
from models.retrieval import ChunkLocation, ChunkSource, EvidenceArtifact, PaperChunk
from models.jobs import WorkflowJob
from models.pdf_uploads import PdfUpload
from models.session import Session, Turn
from storage.models import (
    AgentRunORM,
    ArxivMetadataCacheORM,
    BlobArtifactORM,
    BlobReferenceORM,
    ComparisonArtifactORM,
    PaperChunkORM,
    PaperWorkspaceORM,
    PdfUploadORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
    WorkflowJobORM,
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


def paper_workspace_to_orm(workspace: PaperWorkspace) -> PaperWorkspaceORM:
    return PaperWorkspaceORM(
        id=workspace.id,
        session_id=workspace.session_id,
        paper_id=workspace.paper_id,
        title=workspace.title,
        source_url=workspace.source_url,
        pipeline_stage=workspace.pipeline_stage,
        pipeline_version=workspace.pipeline_version,
        finalized_report_json=workspace.finalized_report_json,
        method_extraction_json=workspace.method_extraction_json,
        benchmarks_json=workspace.benchmarks_json,
        readiness_json=workspace.readiness_json,
        full_markdown_report=workspace.full_markdown_report,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
    )


def orm_to_paper_workspace(orm: PaperWorkspaceORM) -> PaperWorkspace:
    return PaperWorkspace(
        id=orm.id,
        session_id=orm.session_id,
        paper_id=orm.paper_id,
        title=orm.title,
        source_url=orm.source_url,
        pipeline_stage=orm.pipeline_stage,
        pipeline_version=orm.pipeline_version,
        finalized_report_json=orm.finalized_report_json,
        method_extraction_json=orm.method_extraction_json,
        benchmarks_json=list(orm.benchmarks_json or []),
        readiness_json=orm.readiness_json,
        full_markdown_report=orm.full_markdown_report,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def comparison_artifact_to_orm(
    artifact: ComparisonArtifact,
) -> ComparisonArtifactORM:
    return ComparisonArtifactORM(
        id=artifact.id,
        session_id=artifact.session_id,
        paper_ids=artifact.paper_ids,
        comparison_report_json=artifact.comparison_report_json,
        comparison_markdown=artifact.comparison_markdown,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


def orm_to_comparison_artifact(
    orm: ComparisonArtifactORM,
) -> ComparisonArtifact:
    return ComparisonArtifact(
        id=orm.id,
        session_id=orm.session_id,
        paper_ids=list(orm.paper_ids or []),
        comparison_report_json=orm.comparison_report_json,
        comparison_markdown=orm.comparison_markdown,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def arxiv_metadata_cache_entry_to_orm(
    entry: ArxivMetadataCacheEntry,
) -> ArxivMetadataCacheORM:
    return ArxivMetadataCacheORM(
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        authors_json=entry.authors,
        abstract=entry.abstract,
        published_date=entry.published_date,
        categories_json=entry.categories,
        source_url=entry.source_url,
        fetched_at=entry.fetched_at,
        last_error_json=entry.last_error_json,
        error_count=entry.error_count,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


def orm_to_arxiv_metadata_cache_entry(
    orm: ArxivMetadataCacheORM,
) -> ArxivMetadataCacheEntry:
    return ArxivMetadataCacheEntry(
        arxiv_id=orm.arxiv_id,
        title=orm.title,
        authors=list(orm.authors_json or []),
        abstract=orm.abstract,
        published_date=orm.published_date,
        categories=list(orm.categories_json or []),
        source_url=orm.source_url,
        fetched_at=orm.fetched_at,
        last_error_json=orm.last_error_json,
        error_count=orm.error_count,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def blob_artifact_to_orm(artifact: BlobArtifact) -> BlobArtifactORM:
    return BlobArtifactORM(
        id=artifact.id,
        kind=artifact.kind,
        object_key=artifact.object_key,
        bucket_name=artifact.bucket_name,
        content_hash=artifact.content_hash,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        storage_backend=artifact.storage_backend,
        retention_policy=artifact.retention_policy,
        expires_at=artifact.expires_at,
        last_accessed_at=artifact.last_accessed_at,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


def orm_to_blob_artifact(orm: BlobArtifactORM) -> BlobArtifact:
    return BlobArtifact(
        id=orm.id,
        kind=orm.kind,
        object_key=orm.object_key,
        bucket_name=orm.bucket_name,
        content_hash=orm.content_hash,
        content_type=orm.content_type,
        size_bytes=orm.size_bytes,
        storage_backend=orm.storage_backend,
        retention_policy=orm.retention_policy,
        expires_at=orm.expires_at,
        last_accessed_at=orm.last_accessed_at,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def blob_reference_to_orm(reference: BlobReference) -> BlobReferenceORM:
    return BlobReferenceORM(
        id=reference.id,
        blob_id=reference.blob_id,
        ref_kind=reference.ref_kind,
        ref_id=reference.ref_id,
        metadata_json=reference.metadata,
        status=reference.status,
        released_at=reference.released_at,
        created_at=reference.created_at,
    )


def orm_to_blob_reference(orm: BlobReferenceORM) -> BlobReference:
    return BlobReference(
        id=orm.id,
        blob_id=orm.blob_id,
        ref_kind=orm.ref_kind,
        ref_id=orm.ref_id,
        metadata=orm.metadata_json or {},
        status=orm.status,
        released_at=orm.released_at,
        created_at=orm.created_at,
    )


def pdf_upload_to_orm(upload: PdfUpload) -> PdfUploadORM:
    return PdfUploadORM(
        id=upload.id,
        session_id=upload.session_id,
        blob_id=upload.blob_id,
        object_key=upload.object_key,
        expected_sha256=upload.expected_sha256,
        actual_sha256=upload.actual_sha256,
        size_bytes=upload.size_bytes,
        content_type=upload.content_type,
        status=upload.status,
        expires_at=upload.expires_at,
        finalized_at=upload.finalized_at,
        error_json=upload.error_json,
        created_at=upload.created_at,
        updated_at=upload.updated_at,
    )


def orm_to_pdf_upload(orm: PdfUploadORM) -> PdfUpload:
    return PdfUpload(
        id=orm.id,
        session_id=orm.session_id,
        blob_id=orm.blob_id,
        object_key=orm.object_key,
        expected_sha256=orm.expected_sha256,
        actual_sha256=orm.actual_sha256,
        size_bytes=orm.size_bytes,
        content_type=orm.content_type,
        status=orm.status,
        expires_at=orm.expires_at,
        finalized_at=orm.finalized_at,
        error_json=orm.error_json,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def workflow_job_to_orm(job: WorkflowJob) -> WorkflowJobORM:
    return WorkflowJobORM(
        id=job.id,
        session_id=job.session_id,
        kind=job.kind,
        status=job.status,
        input_json=job.input_json or {},
        result_json=job.result_json,
        error_json=job.error_json,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        idempotency_key=job.idempotency_key,
        pipeline_version=job.pipeline_version,
        next_attempt_at=job.next_attempt_at,
        retry_policy_json=job.retry_policy_json,
        locked_by=job.locked_by,
        locked_at=job.locked_at,
        lease_expires_at=job.lease_expires_at,
        heartbeat_at=job.heartbeat_at,
        cancel_requested_at=job.cancel_requested_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def orm_to_workflow_job(orm: WorkflowJobORM) -> WorkflowJob:
    return WorkflowJob(
        id=orm.id,
        session_id=orm.session_id,
        kind=orm.kind,
        status=orm.status,
        input_json=orm.input_json or {},
        result_json=orm.result_json,
        error_json=orm.error_json,
        attempts=orm.attempts,
        max_attempts=orm.max_attempts,
        idempotency_key=orm.idempotency_key,
        pipeline_version=orm.pipeline_version,
        next_attempt_at=orm.next_attempt_at,
        retry_policy_json=orm.retry_policy_json or {},
        locked_by=orm.locked_by,
        locked_at=orm.locked_at,
        lease_expires_at=orm.lease_expires_at,
        heartbeat_at=orm.heartbeat_at,
        cancel_requested_at=orm.cancel_requested_at,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )
