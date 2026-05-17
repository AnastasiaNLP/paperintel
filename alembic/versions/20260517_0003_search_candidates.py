"""search candidates

Revision ID: 20260517_0003
Revises: 20260511_0002
Create Date: 2026-05-17
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260517_0003"
down_revision: str | None = "20260511_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_candidates",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("discovery_turn_id", sa.String(length=64), nullable=False),
        sa.Column("display_rank", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "authors",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("arxiv_id", sa.String(length=64), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column(
            "reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
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
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status in ('proposed', 'selected', 'analyzed', 'rejected')",
            name="ck_search_candidates_status",
        ),
    )
    op.create_index(
        "ix_search_candidates_arxiv_id",
        "search_candidates",
        ["arxiv_id"],
    )
    op.create_index(
        "ix_search_candidates_discovery_turn_id",
        "search_candidates",
        ["discovery_turn_id"],
    )
    op.create_index(
        "ix_search_candidates_session_id",
        "search_candidates",
        ["session_id"],
    )
    op.create_index(
        "ix_search_candidates_source",
        "search_candidates",
        ["source"],
    )
    op.create_index(
        "ix_search_candidates_status",
        "search_candidates",
        ["status"],
    )
    op.create_index(
        "ix_search_candidates_session_turn_rank",
        "search_candidates",
        ["session_id", "discovery_turn_id", "display_rank"],
    )
    op.create_index(
        "ix_search_candidates_session_status",
        "search_candidates",
        ["session_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_search_candidates_session_status", table_name="search_candidates")
    op.drop_index(
        "ix_search_candidates_session_turn_rank",
        table_name="search_candidates",
    )
    op.drop_index("ix_search_candidates_status", table_name="search_candidates")
    op.drop_index("ix_search_candidates_source", table_name="search_candidates")
    op.drop_index("ix_search_candidates_session_id", table_name="search_candidates")
    op.drop_index(
        "ix_search_candidates_discovery_turn_id",
        table_name="search_candidates",
    )
    op.drop_index("ix_search_candidates_arxiv_id", table_name="search_candidates")
    op.drop_table("search_candidates")
