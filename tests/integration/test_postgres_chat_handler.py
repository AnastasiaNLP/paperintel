import os

import pytest
from alembic import command
from alembic.config import Config

from api.app_factory import create_chat_handler
from storage.db import make_engine, make_session_factory
from storage.repositories import clear_foundation_tables


pytestmark = pytest.mark.db


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        return {
            "response_text": "postgres-backed response",
            "intent": "qa",
            "referenced_paper_ids": ["paper-1"],
            "artifact_refs": ["artifact-1"],
            "next_phase": "qa",
        }


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.fixture()
def database_url():
    url = _database_url()
    if not url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for Postgres handler tests")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")

    engine = make_engine(url)
    factory = make_session_factory(engine)
    with factory() as db:
        clear_foundation_tables(db)

    yield url

    with factory() as db:
        clear_foundation_tables(db)
    command.downgrade(config, "base")
    engine.dispose()


def test_chat_handler_uses_postgres_store_and_persistence(database_url):
    runner = FakeRunner()
    handler = create_chat_handler(
        database_url=database_url,
        conversation_runner=runner,
    )

    session = handler.create_session(
        persona="researcher",
        original_query="agent memory",
    )
    result = handler.handle_message(session.id, "answer from stored session")

    assert result.session_id == session.id
    assert result.response_text == "postgres-backed response"
    assert result.phase == "qa"
    assert result.intent == "qa"
    assert result.referenced_paper_ids == ["paper-1"]
    assert result.artifact_refs == ["artifact-1"]

    loaded = handler.store.require_session(session.id)
    assert loaded.phase == "qa"
    assert loaded.persona == "researcher"
    assert loaded.original_query == "agent memory"

    turns = handler.store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "answer from stored session"
    assert turns[1].content == "postgres-backed response"

    call = runner.calls[0]
    assert call["input"]["session_id"] == session.id
    assert call["input"]["phase"] == "idle"
    assert call["config"]["configurable"]["session_id"] == session.id
    assert (
        call["config"]["configurable"]["agent_run_persistence"].__class__.__name__
        == "PostgresAgentRunPersistence"
    )
