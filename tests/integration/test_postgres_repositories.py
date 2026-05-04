import os

import pytest
from alembic import command
from alembic.config import Config

from api.in_memory_session_store import SessionNotFoundError
from models.agent_runs import AgentRun
from models.errors import ErrorCodes, make_error
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresAgentRunPersistence,
    PostgresSessionStore,
    PostgresStructuredErrorRepository,
    clear_foundation_tables,
)


pytestmark = pytest.mark.db


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.fixture()
def session_factory():
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for Postgres repository tests")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    with factory() as db:
        clear_foundation_tables(db)

    yield factory

    with factory() as db:
        clear_foundation_tables(db)
    command.downgrade(config, "base")
    engine.dispose()


def test_postgres_session_store_creates_and_reads_session(session_factory):
    store = PostgresSessionStore(session_factory)

    session = store.create_session(
        persona="researcher",
        original_query="agent memory",
    )

    loaded = store.require_session(session.id)
    assert loaded.id == session.id
    assert loaded.persona == "researcher"
    assert loaded.original_query == "agent memory"
    assert loaded.phase == "idle"


def test_postgres_session_store_updates_phase(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()

    updated = store.update_phase(session.id, "qa")

    assert updated.phase == "qa"
    assert store.require_session(session.id).phase == "qa"


def test_postgres_session_store_appends_and_lists_recent_turns(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()

    store.append_turn(session.id, role="user", content="first")
    store.append_turn(
        session.id,
        role="assistant",
        content="second",
        intent="qa",
        referenced_paper_ids=["paper-1"],
        artifact_refs=["artifact-1"],
        metadata={"source": "test"},
    )

    turns = store.list_recent_turns(session.id)
    assert [turn.content for turn in turns] == ["first", "second"]
    assert turns[1].intent == "qa"
    assert turns[1].referenced_paper_ids == ["paper-1"]
    assert turns[1].artifact_refs == ["artifact-1"]
    assert turns[1].metadata == {"source": "test"}


def test_postgres_session_store_appends_turn_with_structured_error(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    error = make_error(
        ErrorCodes.FATAL_ERROR,
        "graph failed",
        node="chat_handler",
        severity="error",
        recoverable=True,
    )

    turn = store.append_turn(
        session.id,
        role="assistant",
        content="failed",
        error=error,
    )
    turns = store.list_recent_turns(session.id)

    assert turn.error is not None
    assert turn.error.session_id == session.id
    assert turns[0].error is not None
    assert turns[0].error.id == error.id
    assert turns[0].error.message == "graph failed"


def test_postgres_session_store_raises_for_missing_session(session_factory):
    store = PostgresSessionStore(session_factory)

    with pytest.raises(SessionNotFoundError):
        store.require_session("missing")


def test_postgres_agent_run_persistence_upserts_run(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    persistence = PostgresAgentRunPersistence(session_factory)
    run = AgentRun(
        session_id=session.id,
        agent_name="report",
        input_refs=["state:report"],
        model="claude-haiku",
        iteration_count=1,
    )
    run.complete(output_ref="state:report", details={"first": True})

    persistence.save(run)
    run.details["first"] = False
    run.details["second"] = True
    persistence.save(run)

    loaded = persistence.get(run.id)
    assert loaded is not None
    assert loaded.id == run.id
    assert loaded.details == {"first": False, "second": True}


def test_postgres_structured_error_repository_round_trip(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresStructuredErrorRepository(session_factory)
    first = make_error(
        ErrorCodes.WARNING,
        "warning",
        session_id=session.id,
        severity="warning",
        recoverable=True,
    )
    second = make_error(
        ErrorCodes.FATAL_ERROR,
        "fatal",
        session_id=session.id,
        severity="fatal",
        recoverable=False,
    )

    repository.save(first)
    repository.save(second)

    errors = repository.list_for_session(session.id)
    assert [error.id for error in errors] == [first.id, second.id]
    assert [error.message for error in errors] == ["warning", "fatal"]
