import os
from pathlib import Path

import boto3
import pytest
from alembic import command
from alembic.config import Config
from moto import mock_aws
from sqlalchemy import func, select

from models.artifacts import PaperWorkspace
from models.session import HandlerResult
from services.blob_store import S3BlobStore
from services.paperintel_service import PaperIntelService
from storage.db import make_engine, make_session_factory
from storage.models import BlobArtifactORM
from storage.repositories import (
    PostgresBlobArtifactRepository,
    PostgresPaperWorkspaceRepository,
    PostgresSessionStore,
    clear_foundation_tables,
)


pytestmark = pytest.mark.db
BUCKET = "paperintel-blob-ingestion-test"
PDF_BYTES = b"%PDF-1.7\npaperintel durable ingestion\n"


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.fixture()
def stack():
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for blob ingestion tests")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    with session_factory() as db:
        clear_foundation_tables(db)

    session_store = PostgresSessionStore(session_factory)
    workspace_repository = PostgresPaperWorkspaceRepository(session_factory)
    blob_repository = PostgresBlobArtifactRepository(session_factory)
    yield session_store, workspace_repository, blob_repository, session_factory

    with session_factory() as db:
        clear_foundation_tables(db)
    engine.dispose()


class PersistingPdfHandler:
    def __init__(self, *, session_store, workspace_repository) -> None:
        self.store = session_store
        self.workspace_repository = workspace_repository
        self.materialized_paths = []

    def create_session(self, *, persona="engineer", original_query=None):
        return self.store.create_session(
            persona=persona,
            original_query=original_query,
        )

    def analyze_paper_input(
        self,
        session_id,
        *,
        input_type,
        input_value,
        user_content=None,
        expected_paper_id=None,
        skip_arxiv_metadata_fetch=False,
    ):
        materialized_path = Path(input_value)
        assert input_type == "pdf"
        assert materialized_path.exists()
        assert materialized_path.read_bytes() == PDF_BYTES
        self.materialized_paths.append(str(materialized_path))
        self.workspace_repository.upsert_workspace(
            PaperWorkspace(
                session_id=session_id,
                paper_id=expected_paper_id or "local-generated-paper",
                source_url=f"local:{expected_paper_id or 'generated'}",
                pipeline_stage="completed",
            )
        )
        return HandlerResult(
            session_id=session_id,
            response_text="pdf analysis complete",
            phase="qa",
            intent="analyze_paper",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )


@mock_aws
def test_pdf_ingestion_persists_one_blob_and_idempotent_references(stack, tmp_path):
    session_store, workspace_repository, blob_repository, session_factory = stack
    client = boto3.client("s3", region_name="us-east-1")
    blob_store = S3BlobStore(client=client, bucket_name=BUCKET)
    handler = PersistingPdfHandler(
        session_store=session_store,
        workspace_repository=workspace_repository,
    )
    service = PaperIntelService(
        handler=handler,
        artifact_repository=workspace_repository,
        blob_store=blob_store,
        blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    source_path = tmp_path / "paper.pdf"
    source_path.write_bytes(PDF_BYTES)

    service.analyze_pdf(session.id, str(source_path), paper_id="local-paper")
    service.analyze_pdf(session.id, str(source_path), paper_id="local-paper")

    objects = client.list_objects_v2(Bucket=BUCKET)["Contents"]
    assert len(objects) == 1
    artifact = blob_repository.get_by_object_key(objects[0]["Key"])
    assert artifact is not None
    with session_factory() as db:
        artifact_count = db.execute(select(func.count(BlobArtifactORM.id))).scalar_one()
    assert artifact_count == 1
    assert artifact.last_accessed_at is not None
    references = blob_repository.list_references(artifact.id)
    assert [(reference.ref_kind, reference.ref_id) for reference in references] == [
        ("session", session.id),
        (
            "paper_workspace",
            workspace_repository.get_workspace(session.id, "local-paper").id,
        ),
    ]
    assert len(handler.materialized_paths) == 2
    assert all(not Path(path).exists() for path in handler.materialized_paths)
