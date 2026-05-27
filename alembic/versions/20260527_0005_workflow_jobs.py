"""workflow jobs

Revision ID: 20260527_0005
Revises: 20260518_0004
Create Date: 2026-05-27
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260527_0005"
down_revision: str | None = "20260518_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "input_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "result_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "error_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind in ('analyze_paper', 'analyze_selected', 'discover', 'compare', 'synthesize', 'judge_eval')",
            name="ck_workflow_jobs_kind",
        ),
        sa.CheckConstraint(
            "status in ('queued', 'running', 'succeeded', 'failed', 'canceled')",
            name="ck_workflow_jobs_status",
        ),
        sa.CheckConstraint(
            "attempts >= 0",
            name="ck_workflow_jobs_attempts_nonnegative",
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name="ck_workflow_jobs_max_attempts_positive",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_jobs_session_id", "workflow_jobs", ["session_id"])
    op.create_index("ix_workflow_jobs_kind", "workflow_jobs", ["kind"])
    op.create_index("ix_workflow_jobs_status", "workflow_jobs", ["status"])
    op.create_index("ix_workflow_jobs_locked_at", "workflow_jobs", ["locked_at"])
    op.create_index(
        "ix_workflow_jobs_session_created_at",
        "workflow_jobs",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_workflow_jobs_status_created_at",
        "workflow_jobs",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_workflow_jobs_kind_status",
        "workflow_jobs",
        ["kind", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_jobs_kind_status", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_status_created_at", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_session_created_at", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_locked_at", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_status", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_kind", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_session_id", table_name="workflow_jobs")
    op.drop_table("workflow_jobs")
