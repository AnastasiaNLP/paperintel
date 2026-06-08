"""provider rate limits

Revision ID: 20260608_0010
Revises: 20260602_0009
Create Date: 2026-06-08
"""

from typing import Sequence

import sqlalchemy as sa

from alembic import op


revision: str = "20260608_0010"
down_revision: str | None = "20260602_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provider_rate_limits",
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(length=128), nullable=False),
        sa.Column("next_allowed_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.PrimaryKeyConstraint("provider", "operation"),
    )
    op.create_index(
        "ix_provider_rate_limits_next_allowed_at",
        "provider_rate_limits",
        ["next_allowed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_provider_rate_limits_next_allowed_at",
        table_name="provider_rate_limits",
    )
    op.drop_table("provider_rate_limits")
