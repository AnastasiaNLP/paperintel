"""blob artifacts registry

Revision ID: 20260601_0007
Revises: 20260527_0006
Create Date: 2026-06-01
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260601_0007"
down_revision: str | None = "20260527_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "blob_artifacts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("bucket_name", sa.String(length=255), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_backend", sa.String(length=32), nullable=False),
        sa.Column("retention_policy", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
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
            "kind in ('pdf', 'page_image', 'generated_artifact')",
            name="ck_blob_artifacts_kind",
        ),
        sa.CheckConstraint(
            "retention_policy in ('durable', 'ttl')",
            name="ck_blob_artifacts_retention_policy",
        ),
        sa.CheckConstraint(
            "(retention_policy = 'durable' and expires_at is null) "
            "or (retention_policy = 'ttl' and expires_at is not null)",
            name="ck_blob_artifacts_retention_expiry",
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="ck_blob_artifacts_size_nonnegative",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kind",
            "content_hash",
            name="uq_blob_artifacts_kind_content_hash",
        ),
        sa.UniqueConstraint("object_key"),
    )
    op.create_index("ix_blob_artifacts_kind", "blob_artifacts", ["kind"])
    op.create_index(
        "ix_blob_artifacts_expires_at",
        "blob_artifacts",
        ["expires_at"],
    )
    op.create_index(
        "ix_blob_artifacts_last_accessed_at",
        "blob_artifacts",
        ["last_accessed_at"],
    )

    op.create_table(
        "blob_references",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("blob_id", sa.String(length=64), nullable=False),
        sa.Column("ref_kind", sa.String(length=32), nullable=False),
        sa.Column("ref_id", sa.String(length=128), nullable=False),
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
        sa.CheckConstraint(
            "ref_kind in ('session', 'paper_workspace', 'workflow_job')",
            name="ck_blob_references_ref_kind",
        ),
        sa.ForeignKeyConstraint(
            ["blob_id"],
            ["blob_artifacts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "blob_id",
            "ref_kind",
            "ref_id",
            name="uq_blob_references_blob_kind_ref",
        ),
    )
    op.create_index("ix_blob_references_blob_id", "blob_references", ["blob_id"])
    op.create_index("ix_blob_references_ref_kind", "blob_references", ["ref_kind"])
    op.create_index("ix_blob_references_ref_id", "blob_references", ["ref_id"])
    op.create_index(
        "ix_blob_references_kind_ref",
        "blob_references",
        ["ref_kind", "ref_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_blob_references_kind_ref", table_name="blob_references")
    op.drop_index("ix_blob_references_ref_id", table_name="blob_references")
    op.drop_index("ix_blob_references_ref_kind", table_name="blob_references")
    op.drop_index("ix_blob_references_blob_id", table_name="blob_references")
    op.drop_table("blob_references")

    op.drop_index("ix_blob_artifacts_last_accessed_at", table_name="blob_artifacts")
    op.drop_index("ix_blob_artifacts_expires_at", table_name="blob_artifacts")
    op.drop_index("ix_blob_artifacts_kind", table_name="blob_artifacts")
    op.drop_table("blob_artifacts")
