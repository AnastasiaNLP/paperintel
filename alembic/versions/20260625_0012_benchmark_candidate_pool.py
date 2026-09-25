"""benchmark candidate pool

Revision ID: 20260625_0012
Revises: 20260608_0011
Create Date: 2026-06-25
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260625_0012"
down_revision: str | None = "20260608_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "paper_workspaces",
        sa.Column(
            "benchmark_candidates_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "paper_workspaces",
        sa.Column("benchmark_extractor_version", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("paper_workspaces", "benchmark_extractor_version")
    op.drop_column("paper_workspaces", "benchmark_candidates_json")
