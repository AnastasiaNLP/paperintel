"""arxiv metadata cache

Revision ID: 20260527_0006
Revises: 20260527_0005
Create Date: 2026-05-27
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260527_0006"
down_revision: str | None = "20260527_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "arxiv_metadata_cache",
        sa.Column("arxiv_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "authors_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("published_date", sa.String(length=64), nullable=True),
        sa.Column(
            "categories_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_error_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_count", sa.Integer(), nullable=False),
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
            "error_count >= 0",
            name="ck_arxiv_metadata_cache_error_count_nonnegative",
        ),
        sa.PrimaryKeyConstraint("arxiv_id"),
    )
    op.create_index(
        "ix_arxiv_metadata_cache_fetched_at",
        "arxiv_metadata_cache",
        ["fetched_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_arxiv_metadata_cache_fetched_at",
        table_name="arxiv_metadata_cache",
    )
    op.drop_table("arxiv_metadata_cache")
