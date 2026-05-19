import os

import pytest
from alembic import command
from alembic.config import Config

from evaluation.export_workspaces import export_workspaces_for_session
from evaluation.fixtures import build_partial_workspace, build_perfect_workspace
from evaluation.golden_dataset import load_golden_records
from evaluation.runner import load_workspace_records, run_deterministic_evaluation
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresPaperWorkspaceRepository,
    PostgresSessionStore,
    clear_foundation_tables,
)


pytestmark = pytest.mark.db


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.fixture()
def session_factory():
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for evaluation export tests")

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


def test_evaluation_export_flow_uses_persisted_postgres_workspaces(
    session_factory,
    tmp_path,
):
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:2]
    session = PostgresSessionStore(session_factory).create_session(
        persona="engineer",
        original_query="evaluation export",
    )
    repository = PostgresPaperWorkspaceRepository(session_factory)
    first = build_perfect_workspace(records[0]).model_copy(
        update={"session_id": session.id}
    )
    second = build_partial_workspace(records[1]).model_copy(
        update={"session_id": session.id}
    )
    repository.upsert_workspace(first)
    repository.upsert_workspace(second)
    output_path = tmp_path / "workspaces.jsonl"

    exported = export_workspaces_for_session(
        repository=repository,
        session_id=session.id,
        output_path=output_path,
        paper_ids=[records[1].paper_id, records[0].paper_id],
    )
    loaded_workspaces = load_workspace_records(output_path)
    summary = run_deterministic_evaluation(records, loaded_workspaces)

    assert [workspace.paper_id for workspace in exported] == [
        "2005.11401",
        "1706.03762",
    ]
    assert [workspace.paper_id for workspace in loaded_workspaces] == [
        "2005.11401",
        "1706.03762",
    ]
    assert output_path.exists()
    assert summary.total_records == 2
    assert summary.matched_workspaces == 2
    assert summary.missing_workspaces == []
    assert not summary.passed
    assert summary.check_averages["method_extraction"] == 1.0
    assert summary.check_averages["readiness"] == 1.0
    assert 0 < summary.check_averages["benchmarks"] < 1.0

