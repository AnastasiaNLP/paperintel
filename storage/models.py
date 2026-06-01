from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def jsonb_type():
    return JSON().with_variant(postgresql.JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SessionORM(TimestampMixin, Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    persona: Mapped[str] = mapped_column(String(32), nullable=False)
    original_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    phase: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    selected_candidate_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    active_paper_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    latest_comparison_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    turns: Mapped[list["TurnORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    agent_runs: Mapped[list["AgentRunORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    structured_errors: Mapped[list["StructuredErrorORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    search_candidates: Mapped[list["SearchCandidateORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    paper_workspaces: Mapped[list["PaperWorkspaceORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    comparison_artifacts: Mapped[list["ComparisonArtifactORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    workflow_jobs: Mapped[list["WorkflowJobORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class StructuredErrorORM(Base):
    __tablename__ = "structured_errors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    paper_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    agent_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    recoverable: Mapped[bool] = mapped_column(nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[SessionORM | None] = relationship(back_populates="structured_errors")


class TurnORM(Base):
    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    referenced_paper_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    artifact_refs: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    error_id: Mapped[str | None] = mapped_column(
        ForeignKey("structured_errors.id", ondelete="SET NULL"),
        nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[SessionORM] = relationship(back_populates="turns")
    error: Mapped[StructuredErrorORM | None] = relationship()


class AgentRunORM(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    job_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_refs: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    output_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    iteration_count: Mapped[int] = mapped_column(nullable=False, default=0)
    llm_call_count: Mapped[int] = mapped_column(nullable=False, default=0)
    termination_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tokens_used: Mapped[int | None] = mapped_column(nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    session: Mapped[SessionORM | None] = relationship(back_populates="agent_runs")


class ArxivMetadataCacheORM(TimestampMixin, Base):
    __tablename__ = "arxiv_metadata_cache"
    __table_args__ = (
        CheckConstraint(
            "error_count >= 0",
            name="ck_arxiv_metadata_cache_error_count_nonnegative",
        ),
    )

    arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors_json: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    categories_json: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    last_error_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class BlobArtifactORM(TimestampMixin, Base):
    __tablename__ = "blob_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "kind",
            "content_hash",
            name="uq_blob_artifacts_kind_content_hash",
        ),
        CheckConstraint(
            "kind in ('pdf', 'page_image', 'generated_artifact')",
            name="ck_blob_artifacts_kind",
        ),
        CheckConstraint(
            "retention_policy in ('durable', 'ttl')",
            name="ck_blob_artifacts_retention_policy",
        ),
        CheckConstraint(
            "(retention_policy = 'durable' and expires_at is null) "
            "or (retention_policy = 'ttl' and expires_at is not null)",
            name="ck_blob_artifacts_retention_expiry",
        ),
        CheckConstraint(
            "size_bytes >= 0",
            name="ck_blob_artifacts_size_nonnegative",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    object_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    bucket_name: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    references: Mapped[list["BlobReferenceORM"]] = relationship(
        back_populates="blob",
        cascade="all, delete-orphan",
    )


class BlobReferenceORM(Base):
    __tablename__ = "blob_references"
    __table_args__ = (
        UniqueConstraint(
            "blob_id",
            "ref_kind",
            "ref_id",
            name="uq_blob_references_blob_kind_ref",
        ),
        CheckConstraint(
            "ref_kind in ('session', 'paper_workspace', 'workflow_job')",
            name="ck_blob_references_ref_kind",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob_id: Mapped[str] = mapped_column(
        ForeignKey("blob_artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ref_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ref_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    blob: Mapped[BlobArtifactORM] = relationship(back_populates="references")


class WorkflowJobORM(TimestampMixin, Base):
    __tablename__ = "workflow_jobs"
    __table_args__ = (
        CheckConstraint(
            "kind in ('analyze_paper', 'analyze_selected', 'discover', 'compare', 'synthesize', 'judge_eval')",
            name="ck_workflow_jobs_kind",
        ),
        CheckConstraint(
            "status in ('queued', 'running', 'succeeded', 'failed', 'canceled')",
            name="ck_workflow_jobs_status",
        ),
        CheckConstraint("attempts >= 0", name="ck_workflow_jobs_attempts_nonnegative"),
        CheckConstraint("max_attempts >= 1", name="ck_workflow_jobs_max_attempts_positive"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    result_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    error_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    session: Mapped[SessionORM] = relationship(back_populates="workflow_jobs")


class PaperChunkORM(TimestampMixin, Base):
    __tablename__ = "paper_chunks"

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    paper_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    location_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    artifact_refs_json: Mapped[list[dict[str, Any]]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)


class SearchCandidateORM(TimestampMixin, Base):
    __tablename__ = "search_candidates"
    __table_args__ = (
        CheckConstraint(
            "status in ('proposed', 'selected', 'analyzed', 'rejected')",
            name="ck_search_candidates_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    discovery_turn_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    display_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    authors: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    score: Mapped[float | None] = mapped_column(nullable=True)
    reasons: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
        server_default="{}",
    )

    session: Mapped[SessionORM] = relationship(back_populates="search_candidates")


class PaperWorkspaceORM(TimestampMixin, Base):
    __tablename__ = "paper_workspaces"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "paper_id",
            name="uq_paper_workspaces_session_paper",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paper_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    finalized_report_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    method_extraction_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    benchmarks_json: Mapped[list[dict[str, Any]]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    readiness_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    full_markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[SessionORM] = relationship(back_populates="paper_workspaces")


class ComparisonArtifactORM(TimestampMixin, Base):
    __tablename__ = "comparison_artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paper_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
        server_default="[]",
    )
    comparison_report_json: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(),
        nullable=True,
    )
    comparison_markdown: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped[SessionORM] = relationship(back_populates="comparison_artifacts")


Index("ix_turns_session_created_at", TurnORM.session_id, TurnORM.created_at)
Index("ix_agent_runs_session_started_at", AgentRunORM.session_id, AgentRunORM.started_at)
Index("ix_workflow_jobs_session_created_at", WorkflowJobORM.session_id, WorkflowJobORM.created_at)
Index("ix_workflow_jobs_status_created_at", WorkflowJobORM.status, WorkflowJobORM.created_at)
Index("ix_workflow_jobs_kind_status", WorkflowJobORM.kind, WorkflowJobORM.status)
Index("ix_blob_references_kind_ref", BlobReferenceORM.ref_kind, BlobReferenceORM.ref_id)
Index("ix_paper_chunks_paper_chunk", PaperChunkORM.paper_id, PaperChunkORM.chunk_index)
Index("ix_paper_chunks_session_paper", PaperChunkORM.session_id, PaperChunkORM.paper_id)
Index(
    "ix_search_candidates_session_turn_rank",
    SearchCandidateORM.session_id,
    SearchCandidateORM.discovery_turn_id,
    SearchCandidateORM.display_rank,
)
Index(
    "ix_search_candidates_session_status",
    SearchCandidateORM.session_id,
    SearchCandidateORM.status,
)
Index(
    "ix_paper_workspaces_session_created_at",
    PaperWorkspaceORM.session_id,
    PaperWorkspaceORM.created_at,
)
Index(
    "ix_comparison_artifacts_session_created_at",
    ComparisonArtifactORM.session_id,
    ComparisonArtifactORM.created_at,
)
