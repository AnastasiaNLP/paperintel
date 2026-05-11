"""paper chunks

Revision ID: 20260511_0002
Revises: 20260504_0001
Create Date: 2026-05-11
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260511_0002"
down_revision: str | None = "20260504_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.String(length=256), nullable=False),
        sa.Column("paper_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("paper_index", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "source_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "location_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "artifact_refs_json",
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
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_paper_chunks_chunk_type", "paper_chunks", ["chunk_type"])
    op.create_index("ix_paper_chunks_paper_chunk", "paper_chunks", ["paper_id", "chunk_index"])
    op.create_index("ix_paper_chunks_paper_id", "paper_chunks", ["paper_id"])
    op.create_index("ix_paper_chunks_session_id", "paper_chunks", ["session_id"])
    op.create_index("ix_paper_chunks_session_paper", "paper_chunks", ["session_id", "paper_id"])


def downgrade() -> None:
    op.drop_index("ix_paper_chunks_session_paper", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_session_id", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_paper_id", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_paper_chunk", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_chunk_type", table_name="paper_chunks")
    op.drop_table("paper_chunks")
