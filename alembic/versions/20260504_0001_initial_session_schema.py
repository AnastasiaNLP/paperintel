"""initial session schema

Revision ID: 20260504_0001
Revises:
Create Date: 2026-05-04
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260504_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("persona", sa.String(length=32), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=True),
        sa.Column("phase", sa.String(length=32), nullable=False),
        sa.Column(
            "selected_candidate_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "active_paper_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("latest_comparison_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sessions_phase", "sessions", ["phase"])

    op.create_table(
        "structured_errors",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("paper_id", sa.String(length=128), nullable=True),
        sa.Column("job_id", sa.String(length=128), nullable=True),
        sa.Column("agent_run_id", sa.String(length=64), nullable=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("node", sa.String(length=128), nullable=True),
        sa.Column("agent", sa.String(length=128), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("recoverable", sa.Boolean(), nullable=False),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_structured_errors_code", "structured_errors", ["code"])
    op.create_index("ix_structured_errors_session_id", "structured_errors", ["session_id"])
    op.create_index("ix_structured_errors_severity", "structured_errors", ["severity"])

    op.create_table(
        "turns",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=True),
        sa.Column(
            "referenced_paper_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "artifact_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("error_id", sa.String(length=64), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["error_id"], ["structured_errors.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_turns_session_created_at", "turns", ["session_id", "created_at"])
    op.create_index("ix_turns_session_id", "turns", ["session_id"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("job_id", sa.String(length=128), nullable=True),
        sa.Column("agent_name", sa.String(length=128), nullable=False),
        sa.Column(
            "input_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("output_ref", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("iteration_count", sa.Integer(), nullable=False),
        sa.Column("llm_call_count", sa.Integer(), nullable=False),
        sa.Column("termination_reason", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_agent_name", "agent_runs", ["agent_name"])
    op.create_index("ix_agent_runs_job_id", "agent_runs", ["job_id"])
    op.create_index("ix_agent_runs_session_id", "agent_runs", ["session_id"])
    op.create_index("ix_agent_runs_session_started_at", "agent_runs", ["session_id", "started_at"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_session_started_at", table_name="agent_runs")
    op.drop_index("ix_agent_runs_session_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_job_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_agent_name", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index("ix_turns_session_id", table_name="turns")
    op.drop_index("ix_turns_session_created_at", table_name="turns")
    op.drop_table("turns")

    op.drop_index("ix_structured_errors_severity", table_name="structured_errors")
    op.drop_index("ix_structured_errors_session_id", table_name="structured_errors")
    op.drop_index("ix_structured_errors_code", table_name="structured_errors")
    op.drop_table("structured_errors")

    op.drop_index("ix_sessions_phase", table_name="sessions")
    op.drop_table("sessions")
