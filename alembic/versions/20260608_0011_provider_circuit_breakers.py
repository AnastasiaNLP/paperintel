"""provider circuit breakers

Revision ID: 20260608_0011
Revises: 20260608_0010
Create Date: 2026-06-08
"""

from typing import Sequence

import sqlalchemy as sa

from alembic import op


revision: str = "20260608_0011"
down_revision: str | None = "20260608_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provider_circuit_breakers",
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(length=128), nullable=False),
        sa.Column("state", sa.String(length=32), server_default="closed", nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("failure_threshold", sa.Integer(), nullable=False),
        sa.Column("recovery_timeout_seconds", sa.Float(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("open_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("half_open_claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_class", sa.String(length=64), nullable=True),
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
            "state in ('closed', 'open', 'half_open')",
            name="ck_provider_circuit_breakers_state",
        ),
        sa.CheckConstraint(
            "failure_count >= 0",
            name="ck_provider_circuit_breakers_failure_count_nonnegative",
        ),
        sa.PrimaryKeyConstraint("provider", "operation"),
    )
    op.create_index(
        "ix_provider_circuit_breakers_state",
        "provider_circuit_breakers",
        ["state"],
    )
    op.create_index(
        "ix_provider_circuit_breakers_open_until",
        "provider_circuit_breakers",
        ["open_until"],
    )


def downgrade() -> None:
    op.drop_index("ix_provider_circuit_breakers_open_until", table_name="provider_circuit_breakers")
    op.drop_index("ix_provider_circuit_breakers_state", table_name="provider_circuit_breakers")
    op.drop_table("provider_circuit_breakers")
