from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
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


Index("ix_turns_session_created_at", TurnORM.session_id, TurnORM.created_at)
Index("ix_agent_runs_session_started_at", AgentRunORM.session_id, AgentRunORM.started_at)
Index("ix_paper_chunks_paper_chunk", PaperChunkORM.paper_id, PaperChunkORM.chunk_index)
Index("ix_paper_chunks_session_paper", PaperChunkORM.session_id, PaperChunkORM.paper_id)
