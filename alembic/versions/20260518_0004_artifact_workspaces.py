"""artifact workspaces

Revision ID: 20260518_0004
Revises: 20260517_0003
Create Date: 2026-05-18
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260518_0004"
down_revision: str | None = "20260517_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "paper_workspaces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("paper_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("pipeline_stage", sa.String(length=64), nullable=False),
        sa.Column(
            "finalized_report_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "method_extraction_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "benchmarks_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "readiness_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("full_markdown_report", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id",
            "paper_id",
            name="uq_paper_workspaces_session_paper",
        ),
    )
    op.create_index(
        "ix_paper_workspaces_paper_id",
        "paper_workspaces",
        ["paper_id"],
    )
    op.create_index(
        "ix_paper_workspaces_pipeline_stage",
        "paper_workspaces",
        ["pipeline_stage"],
    )
    op.create_index(
        "ix_paper_workspaces_session_created_at",
        "paper_workspaces",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_paper_workspaces_session_id",
        "paper_workspaces",
        ["session_id"],
    )

    op.create_table(
        "comparison_artifacts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column(
            "paper_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "comparison_report_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("comparison_markdown", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_comparison_artifacts_session_created_at",
        "comparison_artifacts",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_comparison_artifacts_session_id",
        "comparison_artifacts",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_comparison_artifacts_session_id",
        table_name="comparison_artifacts",
    )
    op.drop_index(
        "ix_comparison_artifacts_session_created_at",
        table_name="comparison_artifacts",
    )
    op.drop_table("comparison_artifacts")

    op.drop_index("ix_paper_workspaces_session_id", table_name="paper_workspaces")
    op.drop_index(
        "ix_paper_workspaces_session_created_at",
        table_name="paper_workspaces",
    )
    op.drop_index(
        "ix_paper_workspaces_pipeline_stage",
        table_name="paper_workspaces",
    )
    op.drop_index("ix_paper_workspaces_paper_id", table_name="paper_workspaces")
    op.drop_table("paper_workspaces")
