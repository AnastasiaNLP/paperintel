import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


pytestmark = pytest.mark.db


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.mark.skipif(
    not _database_url(),
    reason="PAPERINTEL_TEST_DATABASE_URL is required for Postgres migration smoke",
)
def test_alembic_upgrade_and_downgrade_against_postgres():
    database_url = _database_url()
    assert database_url is not None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    engine = create_engine(database_url, future=True)

    try:
        command.downgrade(config, "base")
        command.upgrade(config, "head")

        inspector = inspect(engine)
        assert {
            "sessions",
            "turns",
            "agent_runs",
            "structured_errors",
            "paper_chunks",
        }.issubset(inspector.get_table_names())

        with engine.connect() as conn:
            version = conn.execute(text("select version_num from alembic_version")).scalar()
        assert version == "20260511_0002"
    finally:
        command.downgrade(config, "base")
        engine.dispose()
