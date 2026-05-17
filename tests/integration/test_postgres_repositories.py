import os
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config

from api.in_memory_session_store import SessionNotFoundError
from models.agent_runs import AgentRun
from models.discovery import SearchCandidate
from models.errors import ErrorCodes, make_error
from models.retrieval import ChunkSource, PaperChunk
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresAgentRunPersistence,
    PostgresPaperChunkRepository,
    PostgresSearchCandidateRepository,
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


def test_postgres_session_store_adds_active_paper(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()

    updated = store.add_active_paper(session.id, "2310.06825")

    assert updated.active_paper_ids == ["2310.06825"]
    assert store.require_session(session.id).active_paper_ids == ["2310.06825"]


def test_postgres_session_store_add_active_paper_is_idempotent(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()

    store.add_active_paper(session.id, "2310.06825")
    store.add_active_paper(session.id, "2310.06825")
    updated = store.add_active_paper(session.id, "2401.12345")

    assert updated.active_paper_ids == ["2310.06825", "2401.12345"]


def test_postgres_session_store_sets_selected_candidate_ids(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()

    updated = store.set_selected_candidate_ids(
        session.id,
        ["candidate-1", "candidate-2", "candidate-1"],
    )

    assert updated.selected_candidate_ids == ["candidate-1", "candidate-2"]
    assert store.require_session(session.id).selected_candidate_ids == [
        "candidate-1",
        "candidate-2",
    ]


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


def test_postgres_paper_chunk_repository_upserts_and_lists_by_paper(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresPaperChunkRepository(session_factory)
    first = PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="Initial retrieval chunk.",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id=session.id,
            arxiv_id="2310.06825",
        ),
    )
    second = PaperChunk(
        id="2310.06825:chunk:1",
        paper_id="2310.06825",
        chunk_index=1,
        text="Second retrieval chunk.",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id=session.id,
            arxiv_id="2310.06825",
        ),
    )

    assert repository.upsert_many([first, second]).model_dump() == {
        "inserted": 2,
        "updated": 0,
        "skipped": 0,
    }

    updated_first = first.model_copy(update={"text": "Updated retrieval chunk."})
    assert repository.upsert_many([updated_first]).model_dump() == {
        "inserted": 0,
        "updated": 1,
        "skipped": 0,
    }

    loaded = repository.list_for_paper("2310.06825")
    assert [chunk.id for chunk in loaded] == [
        "2310.06825:chunk:0",
        "2310.06825:chunk:1",
    ]
    assert loaded[0].text == "Updated retrieval chunk."

    by_ids = repository.get_many_by_ids(
        ["2310.06825:chunk:1", "missing", "2310.06825:chunk:0"]
    )
    assert [chunk.id for chunk in by_ids] == [
        "2310.06825:chunk:1",
        "2310.06825:chunk:0",
    ]


def test_postgres_search_candidate_repository_round_trip(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresSearchCandidateRepository(session_factory)
    first = SearchCandidate(
        session_id=session.id,
        discovery_turn_id="turn-1",
        display_rank=1,
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        arxiv_id="1706.03762",
        published_at=datetime(2017, 6, 12, tzinfo=timezone.utc),
        score=0.95,
        reasons=["exact phrase match"],
    )
    second = SearchCandidate(
        session_id=session.id,
        discovery_turn_id="turn-1",
        display_rank=2,
        title="BERT",
        url="https://arxiv.org/abs/1810.04805",
        arxiv_id="1810.04805",
        score=0.75,
    )

    repository.upsert_many([second, first])

    loaded = repository.list_for_discovery_turn(session.id, "turn-1")
    assert [candidate.id for candidate in loaded] == [first.id, second.id]
    assert loaded[0].status == "proposed"

    updated = repository.update_status(first.id, "selected")
    assert updated is not None
    assert updated.status == "selected"

    latest = repository.list_latest_for_session(session.id)
    assert [candidate.id for candidate in latest] == [first.id, second.id]


def test_postgres_search_candidate_repository_repeated_upsert_preserves_display_ranks(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresSearchCandidateRepository(session_factory)
    first = SearchCandidate(
        session_id=session.id,
        discovery_turn_id="turn-1",
        display_rank=1,
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        arxiv_id="1706.03762",
        score=0.95,
    )
    second = SearchCandidate(
        session_id=session.id,
        discovery_turn_id="turn-1",
        display_rank=2,
        title="BERT",
        url="https://arxiv.org/abs/1810.04805",
        arxiv_id="1810.04805",
        score=0.75,
    )
    batch = [first, second]

    repository.upsert_many(batch)
    repository.upsert_many(batch)

    loaded = repository.list_for_discovery_turn(session.id, "turn-1")
    assert [(candidate.id, candidate.display_rank) for candidate in loaded] == [
        (first.id, 1),
        (second.id, 2),
    ]


def test_postgres_search_candidate_repository_rejects_invalid_status(session_factory):
    repository = PostgresSearchCandidateRepository(session_factory)

    with pytest.raises(ValueError):
        repository.update_status("missing", "invalid")  # type: ignore[arg-type]
