import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, update

from api.in_memory_session_store import SessionNotFoundError
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_storage import StoredBlobObject
from models.discovery import SearchCandidate
from models.external_metadata import ArxivMetadataCacheEntry
from models.jobs import WorkflowJob
from models.pdf_uploads import PdfUpload
from models.registered_pdf_errors import RegisteredPdfBlobNotAuthorizedError
from models.errors import ErrorCodes, make_error
from models.retrieval import ChunkSource, PaperChunk
from storage.db import make_engine, make_session_factory
from storage.models import BlobArtifactORM, BlobReferenceORM, WorkflowJobORM
from storage.repositories import (
    BlobArtifactNotFoundError,
    PostgresAgentRunPersistence,
    PostgresArxivMetadataCacheRepository,
    PostgresBlobArtifactRepository,
    PostgresPaperChunkRepository,
    PostgresPaperWorkspaceRepository,
    PostgresPdfUploadRepository,
    PostgresSearchCandidateRepository,
    PostgresSessionStore,
    PostgresStructuredErrorRepository,
    InvalidWorkflowJobTransitionError,
    InvalidPdfUploadTransitionError,
    PdfUploadExpiredError,
    PostgresWorkflowJobRepository,
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

    by_ids = repository.get_many_by_ids([second.id, "missing", first.id])
    assert [candidate.id for candidate in by_ids] == [second.id, first.id]


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


def test_postgres_paper_workspace_repository_upserts_and_gets_workspace(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresPaperWorkspaceRepository(session_factory)
    workspace = PaperWorkspace(
        session_id=session.id,
        paper_id="1706.03762",
        title="Attention Is All You Need",
        source_url="https://arxiv.org/abs/1706.03762",
        pipeline_stage="chunk_and_index",
        finalized_report_json={"recommended_action": "prototype"},
        method_extraction_json={"method_name": "Transformer"},
        benchmarks_json=[{"task": "translation", "metric": "BLEU"}],
        readiness_json={"maturity_level": "experimental"},
        full_markdown_report="# Report",
    )

    saved = repository.upsert_workspace(workspace)
    updated = repository.upsert_workspace(
        workspace.model_copy(
            update={
                "title": "Transformer",
                "pipeline_stage": "comparison_completed",
                "pipeline_version": "pipeline-v2",
                "finalized_report_json": {"recommended_action": "implement_now"},
                "benchmarks_json": [{"task": "translation", "metric": "accuracy"}],
            }
        )
    )

    loaded = repository.get_workspace(session.id, "1706.03762")
    listed = repository.list_workspaces(session.id)

    assert saved.id == workspace.id
    assert updated.id == workspace.id
    assert loaded is not None
    assert loaded.id == workspace.id
    assert loaded.title == "Transformer"
    assert loaded.pipeline_stage == "comparison_completed"
    assert loaded.pipeline_version == "pipeline-v2"
    assert loaded.finalized_report_json == {"recommended_action": "implement_now"}
    assert loaded.benchmarks_json == [{"task": "translation", "metric": "accuracy"}]
    assert [item.paper_id for item in listed] == ["1706.03762"]


def test_postgres_paper_workspace_repository_scopes_workspaces_by_session(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    first_session = store.create_session()
    second_session = store.create_session()
    repository = PostgresPaperWorkspaceRepository(session_factory)

    repository.upsert_workspace(
        PaperWorkspace(
            session_id=first_session.id,
            paper_id="1706.03762",
            title="First",
            source_url="https://arxiv.org/abs/1706.03762",
            pipeline_stage="chunk_and_index",
        )
    )
    repository.upsert_workspace(
        PaperWorkspace(
            session_id=second_session.id,
            paper_id="1706.03762",
            title="Second",
            source_url="https://arxiv.org/abs/1706.03762",
            pipeline_stage="chunk_and_index",
        )
    )

    first = repository.get_workspace(first_session.id, "1706.03762")
    second = repository.get_workspace(second_session.id, "1706.03762")

    assert first is not None
    assert second is not None
    assert first.title == "First"
    assert second.title == "Second"


def test_postgres_paper_workspace_repository_returns_latest_comparison(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresPaperWorkspaceRepository(session_factory)
    first = ComparisonArtifact(
        session_id=session.id,
        paper_ids=["1706.03762", "1810.04805"],
        comparison_report_json={"winner_basis": "benchmarks"},
        comparison_markdown="# First Comparison",
    )
    second = ComparisonArtifact(
        session_id=session.id,
        paper_ids=["1706.03762", "2605.16113"],
        comparison_report_json={"winner_basis": "readiness"},
        comparison_markdown="# Second Comparison",
    )

    repository.save_comparison(first)
    repository.save_comparison(second)

    latest = repository.latest_comparison(session.id)

    assert latest is not None
    assert latest.id == second.id
    assert latest.paper_ids == ["1706.03762", "2605.16113"]
    assert latest.comparison_markdown == "# Second Comparison"


def test_postgres_paper_workspace_repository_returns_none_for_missing_artifacts(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresPaperWorkspaceRepository(session_factory)

    assert repository.get_workspace(session.id, "missing") is None
    assert repository.latest_comparison(session.id) is None


def test_postgres_workflow_job_repository_lifecycle(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = WorkflowJob(
        session_id=session.id,
        kind="analyze_paper",
        input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
    )

    created = repository.create(job)
    loaded = repository.get(job.id)
    running = repository.mark_running(job.id, worker_id="worker-1")
    succeeded = repository.mark_succeeded(
        job.id,
        worker_id="worker-1",
        result_json={"intent": "analyze_paper", "phase": "qa"},
    )

    assert created.id == job.id
    assert loaded is not None
    assert loaded.input_json == {"paper_url": "https://arxiv.org/abs/1706.03762"}
    assert running.status == "running"
    assert running.locked_by == "worker-1"
    assert running.locked_at is not None
    assert running.started_at is not None
    assert running.attempts == 1
    assert succeeded.status == "succeeded"
    assert succeeded.result_json == {"intent": "analyze_paper", "phase": "qa"}
    assert succeeded.error_json is None
    assert succeeded.finished_at is not None
    assert succeeded.locked_by is None


def test_postgres_workflow_job_repository_failure_and_cancel(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    failed_job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="compare",
            input_json={"paper_ids": ["1706.03762", "2005.11401"]},
        )
    )
    canceled_job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="synthesize",
            input_json={},
        )
    )

    running_failed_job = repository.mark_running(failed_job.id, worker_id="worker-1")
    failed = repository.mark_failed(
        running_failed_job.id,
        worker_id="worker-1",
        error_json={"error": "paper_workspace_not_ready"},
    )
    canceled = repository.mark_canceled(canceled_job.id)

    assert failed.status == "failed"
    assert failed.error_json == {"error": "paper_workspace_not_ready"}
    assert failed.finished_at is not None
    assert canceled.status == "canceled"
    assert canceled.finished_at is not None


def test_postgres_workflow_job_repository_lists_by_session_in_created_order(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    first_session = store.create_session()
    second_session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    base = datetime(2026, 5, 27, tzinfo=timezone.utc)
    first = WorkflowJob(
        session_id=first_session.id,
        kind="discover",
        input_json={"topic": "rag"},
        created_at=base,
    )
    second = WorkflowJob(
        session_id=first_session.id,
        kind="analyze_selected",
        input_json={},
        created_at=base + timedelta(seconds=1),
    )
    other = WorkflowJob(
        session_id=second_session.id,
        kind="discover",
        input_json={"topic": "agents"},
        created_at=base + timedelta(seconds=2),
    )

    repository.create(second)
    repository.create(other)
    repository.create(first)

    listed = repository.list_for_session(first_session.id)

    assert [job.id for job in listed] == [first.id, second.id]


def test_postgres_workflow_job_repository_claims_oldest_queued_job(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    base = datetime(2026, 5, 27, tzinfo=timezone.utc)
    older = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
            created_at=base,
        )
    )
    newer = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="compare",
            input_json={},
            created_at=base + timedelta(seconds=1),
        )
    )

    claimed = repository.claim_next(worker_id="worker-1")
    second_claim = repository.claim_next(worker_id="worker-2")

    assert claimed is not None
    assert claimed.id == older.id
    assert claimed.status == "running"
    assert claimed.locked_by == "worker-1"
    assert claimed.attempts == 1
    assert second_claim is not None
    assert second_claim.id == newer.id
    assert second_claim.locked_by == "worker-2"
    assert repository.claim_next(worker_id="worker-3") is None


def test_postgres_workflow_job_repository_claim_next_filters_by_kind(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    base = datetime(2026, 5, 27, tzinfo=timezone.utc)
    analyze = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
            created_at=base,
        )
    )
    compare = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="compare",
            input_json={},
            created_at=base + timedelta(seconds=1),
        )
    )

    claimed = repository.claim_next(worker_id="worker-1", kinds=["compare"])

    assert claimed is not None
    assert claimed.id == compare.id
    assert repository.get(analyze.id).status == "queued"


def test_postgres_workflow_job_mark_running_is_idempotent_for_same_worker(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
        )
    )

    first = repository.mark_running(job.id, worker_id="worker-1")
    second = repository.mark_running(job.id, worker_id="worker-1")

    assert first.status == "running"
    assert second.status == "running"
    assert first.attempts == 1
    assert second.attempts == 1


def test_postgres_workflow_job_mark_running_rejects_running_job_for_other_worker(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
        )
    )
    repository.mark_running(job.id, worker_id="worker-1")

    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_running(job.id, worker_id="worker-2")


def test_postgres_workflow_job_terminal_transitions_require_running_job(
    session_factory,
):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    queued = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="compare",
            input_json={},
        )
    )

    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_succeeded(queued.id, worker_id="worker-1", result_json={})
    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_failed(queued.id, worker_id="worker-1", error_json={})

    running = repository.mark_running(queued.id, worker_id="worker-1")
    succeeded = repository.mark_succeeded(
        running.id, worker_id="worker-1", result_json={"ok": True}
    )

    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_failed(
            succeeded.id, worker_id="worker-1", error_json={"error": "late"}
        )
    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_canceled(succeeded.id)


def test_postgres_workflow_job_cancel_allows_queued_or_running_only(session_factory):
    store = PostgresSessionStore(session_factory)
    session = store.create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    queued = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="synthesize",
            input_json={},
        )
    )
    running = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="compare",
            input_json={},
        )
    )

    canceled_queued = repository.mark_canceled(queued.id)
    running = repository.mark_running(running.id, worker_id="worker-1")
    cancel_requested = repository.mark_canceled(running.id)
    canceled_running = repository.complete_canceled(running.id, worker_id="worker-1")

    assert canceled_queued.status == "canceled"
    assert cancel_requested.status == "running"
    assert cancel_requested.cancel_requested_at is not None
    assert canceled_running.status == "canceled"
    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_canceled(canceled_queued.id)


def test_postgres_arxiv_metadata_cache_repository_saves_and_reads_global_entry(
    session_factory,
):
    repository = PostgresArxivMetadataCacheRepository(session_factory)
    fetched_at = datetime(2017, 6, 12, tzinfo=timezone.utc)
    entry = ArxivMetadataCacheEntry(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        abstract="Transformer paper.",
        published_date="2017-06-12T17:57:34Z",
        categories=["cs.CL"],
        source_url="https://arxiv.org/abs/1706.03762",
        fetched_at=fetched_at,
    )

    saved = repository.save(entry)
    loaded = repository.get("1706.03762")

    assert saved.arxiv_id == "1706.03762"
    assert loaded is not None
    assert loaded.title == "Attention Is All You Need"
    assert loaded.authors == ["Ashish Vaswani"]
    assert loaded.categories == ["cs.CL"]
    assert loaded.fetched_at == fetched_at
    assert loaded.error_count == 0
    assert repository.get("missing") is None


def test_postgres_arxiv_metadata_cache_repository_records_error_then_success(
    session_factory,
):
    repository = PostgresArxivMetadataCacheRepository(session_factory)

    failed = repository.record_error(
        "2501.12948",
        error_json={"code": "429", "message": "rate limited"},
    )
    failed_again = repository.record_error(
        "2501.12948",
        error_json={"code": "timeout", "message": "timed out"},
    )

    assert failed.error_count == 1
    assert failed_again.error_count == 2
    assert failed_again.last_error_json == {
        "code": "timeout",
        "message": "timed out",
    }
    assert failed_again.has_successful_fetch is False

    first_seen_at = failed_again.created_at
    succeeded = repository.record_success(
        failed_again.model_copy(
            update={
                "title": "Cached title",
                "authors": [],
                "abstract": "Cached abstract.",
                "published_date": "2025-01",
                "categories": ["cs.CL"],
                "source_url": "https://arxiv.org/abs/2501.12948",
            }
        )
    )

    assert succeeded.created_at == first_seen_at
    assert succeeded.error_count == 0
    assert succeeded.last_error_json is None
    assert succeeded.has_successful_fetch is True


def _stored_pdf(*, content_hash: str = "a" * 64) -> StoredBlobObject:
    return StoredBlobObject(
        kind="pdf",
        object_key=f"papers/sha256/{content_hash[:2]}/{content_hash}.pdf",
        bucket_name="paperintel",
        content_hash=content_hash,
        content_type="application/pdf",
        size_bytes=128,
    )


def test_postgres_blob_artifact_repository_upserts_and_preserves_created_at(
    session_factory,
):
    repository = PostgresBlobArtifactRepository(session_factory)
    first = repository.upsert_artifact(_stored_pdf())
    second = repository.upsert_artifact(
        _stored_pdf().model_copy(update={"size_bytes": 256})
    )

    assert second.id == first.id
    assert second.created_at == first.created_at
    assert second.size_bytes == 256
    assert repository.get_artifact(first.id) == second
    assert repository.get_by_kind_and_hash("pdf", "a" * 64) == second
    assert repository.get_by_object_key(second.object_key) == second


def test_postgres_blob_artifact_repository_references_are_idempotent(session_factory):
    repository = PostgresBlobArtifactRepository(session_factory)
    artifact = repository.upsert_artifact(_stored_pdf())

    first = repository.add_reference(
        artifact.id,
        ref_kind="session",
        ref_id="session-1",
        metadata={"source": "upload"},
    )
    second = repository.add_reference(
        artifact.id,
        ref_kind="session",
        ref_id="session-1",
        metadata={"source": "duplicate"},
    )
    workspace = repository.add_reference(
        artifact.id,
        ref_kind="paper_workspace",
        ref_id="workspace-1",
    )

    assert second == first
    assert workspace.id != first.id
    assert repository.list_references(artifact.id) == [first, workspace]
    assert repository.has_active_reference(
        artifact.id, ref_kind="session", ref_id="session-1"
    ) is True
    assert repository.has_active_reference(
        artifact.id, ref_kind="session", ref_id="missing"
    ) is False
    assert repository.list_artifacts_for_reference(
        ref_kind="session",
        ref_id="session-1",
    ) == [artifact]

    repository.release_reference(
        artifact.id,
        ref_kind="session",
        ref_id="session-1",
    )
    references = repository.list_references(artifact.id)
    assert [(reference.id, reference.status) for reference in references] == [
        (first.id, "released"),
        (workspace.id, "active"),
    ]
    assert references[0].released_at is not None
    assert repository.has_active_reference(
        artifact.id, ref_kind="session", ref_id="session-1"
    ) is False
    assert repository.list_artifacts_for_reference(
        ref_kind="session", ref_id="session-1"
    ) == []


def test_postgres_blob_artifact_repository_marks_material_access(session_factory):
    repository = PostgresBlobArtifactRepository(session_factory)
    artifact = repository.upsert_artifact(_stored_pdf())

    accessed = repository.mark_accessed(artifact.id)

    assert artifact.last_accessed_at is None
    assert accessed.last_accessed_at is not None
    assert accessed.updated_at == artifact.updated_at
    assert repository.get_artifact(artifact.id).last_accessed_at == accessed.last_accessed_at


def test_postgres_blob_artifact_repository_rejects_reference_for_missing_blob(
    session_factory,
):
    repository = PostgresBlobArtifactRepository(session_factory)

    with pytest.raises(BlobArtifactNotFoundError):
        repository.add_reference(
            "missing",
            ref_kind="session",
            ref_id="session-1",
        )

    with pytest.raises(BlobArtifactNotFoundError):
        repository.mark_accessed("missing")


def test_postgres_blob_reference_cascades_when_artifact_is_deleted(session_factory):
    repository = PostgresBlobArtifactRepository(session_factory)
    artifact = repository.upsert_artifact(_stored_pdf())
    reference = repository.add_reference(
        artifact.id,
        ref_kind="workflow_job",
        ref_id="job-1",
    )

    with session_factory() as db:
        db.execute(delete(BlobArtifactORM).where(BlobArtifactORM.id == artifact.id))
        db.commit()

    with session_factory() as db:
        assert db.get(BlobReferenceORM, reference.id) is None


def test_postgres_blob_artifact_repository_concurrent_upsert_is_idempotent(
    session_factory,
):
    barrier = Barrier(2)

    def upsert_once():
        repository = PostgresBlobArtifactRepository(session_factory)
        barrier.wait()
        return repository.upsert_artifact(_stored_pdf())

    with ThreadPoolExecutor(max_workers=2) as pool:
        artifacts = list(pool.map(lambda _: upsert_once(), range(2)))

    repository = PostgresBlobArtifactRepository(session_factory)
    assert len({artifact.id for artifact in artifacts}) == 1
    persisted = repository.get_by_kind_and_hash("pdf", "a" * 64)
    assert persisted is not None
    assert persisted.id == artifacts[0].id


def test_postgres_blob_artifact_repository_concurrent_reference_is_idempotent(
    session_factory,
):
    repository = PostgresBlobArtifactRepository(session_factory)
    artifact = repository.upsert_artifact(_stored_pdf())
    barrier = Barrier(2)

    def add_once():
        thread_repository = PostgresBlobArtifactRepository(session_factory)
        barrier.wait()
        return thread_repository.add_reference(
            artifact.id,
            ref_kind="session",
            ref_id="session-1",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        references = list(pool.map(lambda _: add_once(), range(2)))

    assert len({reference.id for reference in references}) == 1
    assert repository.list_references(artifact.id) == [references[0]]


def test_postgres_pdf_upload_repository_enforces_lifecycle(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    artifact = blob_repository.upsert_artifact(_stored_pdf())
    repository = PostgresPdfUploadRepository(session_factory)
    digest = "a" * 64
    upload = repository.create(
        PdfUpload(
            session_id=session.id,
            object_key=f"uploads/{session.id}/upload-1.pdf",
            expected_sha256=digest,
            size_bytes=128,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )

    uploaded = repository.mark_uploaded(upload.id)
    finalized = repository.finalize(
        upload.id, blob_id=artifact.id, actual_sha256=digest, size_bytes=128
    )
    enqueued = repository.mark_enqueued(upload.id)

    assert uploaded.status == "uploaded"
    assert finalized.status == "finalized"
    assert finalized.blob_id == artifact.id
    assert finalized.actual_sha256 == digest
    assert enqueued.status == "enqueued"
    with pytest.raises(InvalidPdfUploadTransitionError):
        repository.mark_uploaded(upload.id)


def test_postgres_pdf_upload_repository_rejects_expired_finalize(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresPdfUploadRepository(session_factory)
    upload = repository.create(
        PdfUpload(
            session_id=session.id,
            object_key=f"uploads/{session.id}/expired.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )

    repository.mark_uploaded(upload.id)
    with pytest.raises(PdfUploadExpiredError):
        repository.finalize(
            upload.id, blob_id="unused", actual_sha256="a" * 64, size_bytes=128
        )


def test_postgres_pdf_upload_repository_serializes_competing_transitions(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresPdfUploadRepository(session_factory)
    upload = repository.create(
        PdfUpload(
            session_id=session.id, object_key=f"uploads/{session.id}/race.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    barrier = Barrier(2)

    def transition_once():
        thread_repository = PostgresPdfUploadRepository(session_factory)
        barrier.wait()
        try:
            return thread_repository.mark_uploaded(upload.id).status
        except InvalidPdfUploadTransitionError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: transition_once(), range(2)))

    assert sorted(statuses) == ["rejected", "uploaded"]
    assert repository.get(upload.id).status == "uploaded"


def test_postgres_pdf_upload_repository_allows_only_one_terminal_transition(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    artifact = blob_repository.upsert_artifact(_stored_pdf())
    repository = PostgresPdfUploadRepository(session_factory)
    upload = repository.create(
        PdfUpload(
            session_id=session.id, object_key=f"uploads/{session.id}/terminal-race.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    repository.mark_uploaded(upload.id)
    barrier = Barrier(2)

    def finalize_once():
        thread_repository = PostgresPdfUploadRepository(session_factory)
        barrier.wait()
        try:
            return thread_repository.finalize(
                upload.id, blob_id=artifact.id, actual_sha256="a" * 64, size_bytes=128
            ).status
        except InvalidPdfUploadTransitionError:
            return "rejected"

    def fail_once():
        thread_repository = PostgresPdfUploadRepository(session_factory)
        barrier.wait()
        try:
            return thread_repository.mark_failed(
                upload.id, error_json={"code": "test_failure"}
            ).status
        except InvalidPdfUploadTransitionError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = [pool.submit(finalize_once), pool.submit(fail_once)]
        statuses = [future.result() for future in statuses]

    assert statuses.count("rejected") == 1
    assert repository.get(upload.id).status in {"finalized", "failed"}


def test_postgres_workflow_job_repository_enqueues_pdf_blob_idempotently(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    artifact = blob_repository.upsert_artifact(_stored_pdf())
    blob_repository.add_reference(artifact.id, ref_kind="session", ref_id=session.id)
    upload_repository = PostgresPdfUploadRepository(session_factory)
    upload = upload_repository.create(
        PdfUpload(
            session_id=session.id,
            object_key=f"uploads/{session.id}/enqueue.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    upload_repository.mark_uploaded(upload.id)
    upload_repository.finalize(
        upload.id, blob_id=artifact.id, actual_sha256="a" * 64, size_bytes=128
    )
    repository = PostgresWorkflowJobRepository(session_factory)
    first = repository.enqueue_pdf_blob(
        session_id=session.id,
        upload_id=upload.id,
        paper_id="paper-1",
        skip_arxiv_metadata_fetch=False,
        pipeline_version="v1",
    )
    second = repository.enqueue_pdf_blob(
        session_id=session.id,
        upload_id=upload.id,
        paper_id="paper-1",
        skip_arxiv_metadata_fetch=False,
        pipeline_version="v1",
    )

    assert second.id == first.id
    assert upload_repository.get(upload.id).status == "enqueued"
    references = blob_repository.list_references(artifact.id)
    assert [(reference.ref_kind, reference.ref_id, reference.status) for reference in references] == [
        ("session", session.id, "active"),
        ("workflow_job", first.id, "active"),
    ]

    running = repository.claim_next(worker_id="worker-1")
    retry = repository.record_failure(
        running.id,
        worker_id="worker-1",
        error_json={"error": "blob_store_unavailable"},
        retryable=True,
    )
    assert retry.status == "queued"
    references = blob_repository.list_references(artifact.id)
    assert references[1].status == "active"

    canceled = repository.mark_canceled(first.id)
    assert canceled.status == "canceled"
    references = blob_repository.list_references(artifact.id)
    assert [(reference.ref_kind, reference.ref_id, reference.status) for reference in references] == [
        ("session", session.id, "active"),
        ("workflow_job", first.id, "released"),
    ]
    assert references[1].released_at is not None


def test_postgres_workflow_job_repository_serializes_pdf_enqueue_race(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    artifact = blob_repository.upsert_artifact(_stored_pdf())
    blob_repository.add_reference(artifact.id, ref_kind="session", ref_id=session.id)
    upload_repository = PostgresPdfUploadRepository(session_factory)
    upload = upload_repository.create(
        PdfUpload(
            session_id=session.id,
            object_key=f"uploads/{session.id}/enqueue-race.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    upload_repository.mark_uploaded(upload.id)
    upload_repository.finalize(
        upload.id, blob_id=artifact.id, actual_sha256="a" * 64, size_bytes=128
    )
    barrier = Barrier(2)

    def enqueue_once(index):
        barrier.wait()
        return PostgresWorkflowJobRepository(session_factory).enqueue_pdf_blob(
            session_id=session.id,
            upload_id=upload.id,
            paper_id="paper-1",
            skip_arxiv_metadata_fetch=False,
            pipeline_version="v1",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(enqueue_once, range(2)))

    assert len({job.id for job in jobs}) == 1
    references = blob_repository.list_references(artifact.id)
    assert [(reference.ref_kind, reference.ref_id) for reference in references] == [
        ("session", session.id),
        ("workflow_job", jobs[0].id),
    ]


@pytest.mark.parametrize("status", ["initiated", "uploaded", "failed"])
def test_postgres_workflow_job_repository_rejects_unfinalized_pdf_upload(
    session_factory, status
):
    session = PostgresSessionStore(session_factory).create_session()
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    artifact = blob_repository.upsert_artifact(_stored_pdf())
    blob_repository.add_reference(artifact.id, ref_kind="session", ref_id=session.id)
    upload = PostgresPdfUploadRepository(session_factory).create(
        PdfUpload(
            session_id=session.id,
            blob_id=artifact.id,
            object_key=f"uploads/{session.id}/{status}.pdf",
            expected_sha256="a" * 64,
            status=status,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )

    with pytest.raises(InvalidPdfUploadTransitionError):
        PostgresWorkflowJobRepository(session_factory).enqueue_pdf_blob(
            session_id=session.id,
            upload_id=upload.id,
            paper_id=None,
            skip_arxiv_metadata_fetch=False,
            pipeline_version="v1",
        )


def test_postgres_workflow_job_repository_rolls_back_unauthorized_pdf_enqueue(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    artifact = PostgresBlobArtifactRepository(session_factory).upsert_artifact(
        _stored_pdf()
    )
    upload_repository = PostgresPdfUploadRepository(session_factory)
    upload = upload_repository.create(
        PdfUpload(
            session_id=session.id,
            object_key=f"uploads/{session.id}/unauthorized.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    upload_repository.mark_uploaded(upload.id)
    upload_repository.finalize(
        upload.id, blob_id=artifact.id, actual_sha256="a" * 64, size_bytes=128
    )
    repository = PostgresWorkflowJobRepository(session_factory)

    with pytest.raises(RegisteredPdfBlobNotAuthorizedError):
        repository.enqueue_pdf_blob(
            session_id=session.id,
            upload_id=upload.id,
            paper_id=None,
            skip_arxiv_metadata_fetch=False,
            pipeline_version="v1",
        )

    assert upload_repository.get(upload.id).status == "finalized"
    assert repository.list_for_session(session.id) == []


def test_postgres_workflow_job_repository_heartbeat_requires_current_owner(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(session_id=session.id, kind="analyze_paper", input_json={})
    )
    with pytest.raises(ValueError, match="lease_seconds must be positive"):
        repository.mark_running(job.id, worker_id="worker-1", lease_seconds=0)
    running = repository.claim_next(worker_id="worker-1", lease_seconds=30)

    renewed = repository.heartbeat(
        running.id, worker_id="worker-1", lease_seconds=60
    )

    assert renewed.heartbeat_at >= running.heartbeat_at
    assert renewed.lease_expires_at > running.lease_expires_at
    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.heartbeat(running.id, worker_id="worker-2", lease_seconds=60)


def test_postgres_workflow_job_repository_reclaims_expired_lease_and_rejects_stale_owner(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
            max_attempts=3,
        )
    )
    repository.claim_next(worker_id="worker-1")
    with session_factory() as db:
        db.execute(
            update(WorkflowJobORM)
            .where(WorkflowJobORM.id == job.id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
        db.commit()

    reclaimed = repository.claim_next(worker_id="worker-2")

    assert reclaimed.id == job.id
    assert reclaimed.status == "running"
    assert reclaimed.locked_by == "worker-2"
    assert reclaimed.attempts == 2
    with pytest.raises(InvalidWorkflowJobTransitionError):
        repository.mark_succeeded(job.id, worker_id="worker-1", result_json={})


def test_postgres_workflow_job_repository_reclaims_legacy_running_job_without_lease(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            status="running",
            input_json={},
            attempts=1,
            max_attempts=3,
            locked_by="legacy-worker",
        )
    )

    reclaimed = repository.claim_next(worker_id="worker-2")

    assert reclaimed.id == job.id
    assert reclaimed.status == "running"
    assert reclaimed.locked_by == "worker-2"
    assert reclaimed.attempts == 2
    assert reclaimed.lease_expires_at is not None


def test_postgres_workflow_job_repository_exhausts_reclaimed_job_without_execution(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
            max_attempts=2,
        )
    )
    repository.claim_next(worker_id="worker-1")
    with session_factory() as db:
        db.execute(
            update(WorkflowJobORM)
            .where(WorkflowJobORM.id == job.id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
        db.commit()

    exhausted = repository.claim_next(worker_id="worker-2")

    assert exhausted.status == "failed"
    assert exhausted.error_json["error"] == "retry_exhausted"
    assert exhausted.locked_by is None


def test_postgres_workflow_job_repository_cancels_reclaimed_job_before_retry_exhaustion(
    session_factory,
):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
            max_attempts=2,
        )
    )
    running = repository.claim_next(worker_id="worker-1")
    repository.mark_canceled(running.id)
    with session_factory() as db:
        db.execute(
            update(WorkflowJobORM)
            .where(WorkflowJobORM.id == job.id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
        db.commit()

    canceled = repository.claim_next(worker_id="worker-2")

    assert canceled.status == "canceled"
    assert canceled.error_json["error"] == "job_canceled"
    assert canceled.locked_by is None


def test_postgres_workflow_job_repository_retry_respects_schedule(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(
            session_id=session.id,
            kind="analyze_paper",
            input_json={},
            max_attempts=3,
            retry_policy_json={"base_delay_seconds": 60},
        )
    )
    running = repository.claim_next(worker_id="worker-1")

    retry = repository.record_failure(
        running.id,
        worker_id="worker-1",
        error_json={"error": "provider_timeout"},
        retryable=True,
    )

    assert retry.status == "queued"
    assert retry.next_attempt_at is not None
    assert retry.locked_by is None
    assert repository.claim_next(worker_id="worker-2") is None


def test_postgres_workflow_job_success_commit_honors_cancel_request(session_factory):
    session = PostgresSessionStore(session_factory).create_session()
    repository = PostgresWorkflowJobRepository(session_factory)
    job = repository.create(
        WorkflowJob(session_id=session.id, kind="analyze_paper", input_json={})
    )
    running = repository.claim_next(worker_id="worker-1")
    requested = repository.mark_canceled(running.id)

    terminal = repository.mark_succeeded(
        running.id, worker_id="worker-1", result_json={"unexpected": True}
    )

    assert requested.status == "running"
    assert terminal.status == "canceled"
    assert terminal.result_json is None
    assert terminal.error_json["error"] == "job_canceled"
