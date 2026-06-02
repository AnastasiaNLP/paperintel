"""async PDF upload and job reliability foundation

Revision ID: 20260602_0008
Revises: 20260601_0007
Create Date: 2026-06-02
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260602_0008"
down_revision: str | None = "20260601_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "blob_references",
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
    )
    op.add_column(
        "blob_references",
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_blob_references_status",
        "blob_references",
        "status in ('active', 'released')",
    )
    op.create_check_constraint(
        "ck_blob_references_release_state",
        "blob_references",
        "(status = 'active' and released_at is null) "
        "or (status = 'released' and released_at is not null)",
    )
    op.create_index(
        "ix_blob_references_status",
        "blob_references",
        ["status"],
    )

    op.drop_constraint("ck_workflow_jobs_kind", "workflow_jobs", type_="check")
    op.create_check_constraint(
        "ck_workflow_jobs_kind",
        "workflow_jobs",
        "kind in ('analyze_paper', 'analyze_selected', 'analyze_pdf_blob', "
        "'discover', 'compare', 'synthesize', 'judge_eval')",
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("pipeline_version", sa.String(length=128), server_default="v1", nullable=False),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column(
            "retry_policy_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_jobs",
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint(
        "uq_workflow_jobs_idempotency_key",
        "workflow_jobs",
        ["idempotency_key"],
    )
    op.create_index(
        "ix_workflow_jobs_next_attempt_at",
        "workflow_jobs",
        ["next_attempt_at"],
    )
    op.create_index(
        "ix_workflow_jobs_lease_expires_at",
        "workflow_jobs",
        ["lease_expires_at"],
    )

    op.add_column(
        "paper_workspaces",
        sa.Column("pipeline_version", sa.String(length=128), server_default="v1", nullable=False),
    )

    op.create_table(
        "pdf_uploads",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("blob_id", sa.String(length=64), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("expected_sha256", sa.String(length=64), nullable=True),
        sa.Column("actual_sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "error_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
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
            "status in ('initiated', 'uploaded', 'finalized', 'enqueued', 'failed', 'expired')",
            name="ck_pdf_uploads_status",
        ),
        sa.CheckConstraint(
            "size_bytes is null or size_bytes >= 0",
            name="ck_pdf_uploads_size_nonnegative",
        ),
        sa.CheckConstraint(
            "status not in ('finalized', 'enqueued') or "
            "(blob_id is not null and expected_sha256 is not null and "
            "actual_sha256 is not null and expected_sha256 = actual_sha256 "
            "and finalized_at is not null)",
            name="ck_pdf_uploads_finalized_integrity",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["blob_id"], ["blob_artifacts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("object_key", name="uq_pdf_uploads_object_key"),
    )
    op.create_index("ix_pdf_uploads_session_id", "pdf_uploads", ["session_id"])
    op.create_index("ix_pdf_uploads_blob_id", "pdf_uploads", ["blob_id"])
    op.create_index("ix_pdf_uploads_status", "pdf_uploads", ["status"])
    op.create_index("ix_pdf_uploads_expires_at", "pdf_uploads", ["expires_at"])
    op.create_index(
        "ix_pdf_uploads_session_created_at",
        "pdf_uploads",
        ["session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_pdf_uploads_session_created_at", table_name="pdf_uploads")
    op.drop_index("ix_pdf_uploads_expires_at", table_name="pdf_uploads")
    op.drop_index("ix_pdf_uploads_status", table_name="pdf_uploads")
    op.drop_index("ix_pdf_uploads_blob_id", table_name="pdf_uploads")
    op.drop_index("ix_pdf_uploads_session_id", table_name="pdf_uploads")
    op.drop_table("pdf_uploads")

    op.drop_column("paper_workspaces", "pipeline_version")

    op.drop_index("ix_workflow_jobs_lease_expires_at", table_name="workflow_jobs")
    op.drop_index("ix_workflow_jobs_next_attempt_at", table_name="workflow_jobs")
    op.drop_constraint("uq_workflow_jobs_idempotency_key", "workflow_jobs", type_="unique")
    op.drop_column("workflow_jobs", "cancel_requested_at")
    op.drop_column("workflow_jobs", "heartbeat_at")
    op.drop_column("workflow_jobs", "lease_expires_at")
    op.drop_column("workflow_jobs", "retry_policy_json")
    op.drop_column("workflow_jobs", "next_attempt_at")
    op.drop_column("workflow_jobs", "pipeline_version")
    op.drop_column("workflow_jobs", "idempotency_key")
    op.drop_constraint("ck_workflow_jobs_kind", "workflow_jobs", type_="check")
    op.execute("delete from workflow_jobs where kind = 'analyze_pdf_blob'")
    op.create_check_constraint(
        "ck_workflow_jobs_kind",
        "workflow_jobs",
        "kind in ('analyze_paper', 'analyze_selected', 'discover', 'compare', "
        "'synthesize', 'judge_eval')",
    )

    op.drop_index("ix_blob_references_status", table_name="blob_references")
    op.drop_constraint("ck_blob_references_release_state", "blob_references", type_="check")
    op.drop_constraint("ck_blob_references_status", "blob_references", type_="check")
    op.drop_column("blob_references", "released_at")
    op.drop_column("blob_references", "status")
