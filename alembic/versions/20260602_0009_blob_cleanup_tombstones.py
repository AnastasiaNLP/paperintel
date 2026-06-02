"""blob cleanup tombstones

Revision ID: 20260602_0009
Revises: 20260602_0008
Create Date: 2026-06-02
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260602_0009"
down_revision: str | None = "20260602_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "blob_artifacts",
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
    )
    op.add_column(
        "blob_artifacts",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "blob_artifacts",
        sa.Column(
            "cleanup_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_blob_artifacts_status",
        "blob_artifacts",
        "status in ('active', 'deleted')",
    )
    op.create_check_constraint(
        "ck_blob_artifacts_deletion_state",
        "blob_artifacts",
        "(status = 'active' and deleted_at is null) "
        "or (status = 'deleted' and deleted_at is not null)",
    )
    op.create_index("ix_blob_artifacts_status", "blob_artifacts", ["status"])
    op.create_index("ix_blob_artifacts_deleted_at", "blob_artifacts", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_blob_artifacts_deleted_at", table_name="blob_artifacts")
    op.drop_index("ix_blob_artifacts_status", table_name="blob_artifacts")
    op.drop_constraint("ck_blob_artifacts_deletion_state", "blob_artifacts", type_="check")
    op.drop_constraint("ck_blob_artifacts_status", "blob_artifacts", type_="check")
    op.drop_column("blob_artifacts", "cleanup_metadata_json")
    op.drop_column("blob_artifacts", "deleted_at")
    op.drop_column("blob_artifacts", "status")
