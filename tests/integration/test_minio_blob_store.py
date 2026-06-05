import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from api.app_factory import _resolve_blob_store
from models.artifacts import PaperWorkspace
from models.pdf_uploads import PdfUpload
from models.session import HandlerResult
from services.blob_cleanup import BlobCleanupService
from services.blob_store import S3BlobStore
from services.paperintel_service import PaperIntelService
from storage.db import make_engine, make_session_factory
from storage.models import BlobArtifactORM
from storage.repositories import (
    PostgresBlobArtifactRepository,
    PostgresBlobCleanupRepository,
    PostgresPaperWorkspaceRepository,
    PostgresPdfUploadRepository,
    PostgresSessionStore,
    clear_foundation_tables,
)


pytestmark = pytest.mark.db
PDF_BYTES = b"%PDF-1.7\npaperintel real minio integration\n"
PNG_BYTES = b"paperintel real minio cleanup png\n"


def _minio_url() -> str | None:
    return os.environ.get("PAPERINTEL_MINIO_TEST_URL")


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


def _blob_settings():
    return SimpleNamespace(
        blob_storage_enabled=True,
        blob_s3_endpoint_url=_minio_url(),
        blob_s3_region="us-east-1",
        blob_s3_bucket=os.environ.get("PAPERINTEL_MINIO_TEST_BUCKET", "paperintel-test"),
        blob_s3_access_key_id=os.environ.get(
            "PAPERINTEL_MINIO_TEST_ACCESS_KEY_ID",
            "paperintel",
        ),
        blob_s3_secret_access_key=os.environ.get(
            "PAPERINTEL_MINIO_TEST_SECRET_ACCESS_KEY",
            "paperintel_dev_password",
        ),
    )


def _store() -> S3BlobStore:
    minio_url = _minio_url()
    if not minio_url:
        pytest.skip("PAPERINTEL_MINIO_TEST_URL is required for MinIO integration tests")
    settings = _blob_settings()
    return S3BlobStore.from_config(
        bucket_name=settings.blob_s3_bucket,
        endpoint_url=settings.blob_s3_endpoint_url,
        region_name=settings.blob_s3_region,
        access_key_id=settings.blob_s3_access_key_id,
        secret_access_key=settings.blob_s3_secret_access_key,
    )


def test_real_minio_blob_store_round_trip_and_dedup():
    store = _store()
    store.ensure_bucket()
    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    object_key = f"papers/sha256/{expected_hash[:2]}/{expected_hash}.pdf"

    try:
        first = store.put(PDF_BYTES, kind="pdf")
        second = store.put(PDF_BYTES, kind="pdf")

        assert first == second
        assert first.content_hash == expected_hash
        assert first.object_key == object_key
        assert store.exists(first.object_key)

        with store.materialize(
            first.object_key,
            expected_sha256=expected_hash,
        ) as path:
            materialized = Path(path)
            assert materialized.read_bytes() == PDF_BYTES

        assert not materialized.exists()
    finally:
        if store.exists(object_key):
            store.delete(object_key)

    assert not store.exists(object_key)


class PersistingMinioPdfHandler:
    def __init__(self, *, session_store, workspace_repository) -> None:
        self.store = session_store
        self.workspace_repository = workspace_repository
        self.materialized_paths = []

    def create_session(self, *, persona="engineer", original_query=None):
        return self.store.create_session(persona=persona, original_query=original_query)

    def analyze_paper_input(
        self,
        session_id,
        *,
        input_type,
        input_value,
        user_content=None,
        expected_paper_id=None,
        skip_arxiv_metadata_fetch=False,
        pipeline_version="v1",
    ):
        materialized_path = Path(input_value)
        assert input_type == "pdf"
        assert materialized_path.read_bytes() == PDF_BYTES
        self.materialized_paths.append(str(materialized_path))
        self.workspace_repository.upsert_workspace(
            PaperWorkspace(
                session_id=session_id,
                paper_id=expected_paper_id or "local-generated-paper",
                source_url=f"local:{expected_paper_id or 'generated'}",
                pipeline_stage="completed",
                pipeline_version=pipeline_version,
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


def test_real_minio_and_postgres_durable_pdf_ingestion(tmp_path):
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for MinIO ingestion test")
    if not _minio_url():
        pytest.skip("PAPERINTEL_MINIO_TEST_URL is required for MinIO ingestion test")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    blob_store = None
    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    object_key = f"papers/sha256/{expected_hash[:2]}/{expected_hash}.pdf"

    try:
        with session_factory() as db:
            clear_foundation_tables(db)

        blob_store = _resolve_blob_store(
            blob_store=None,
            enable_blob_storage=None,
            settings=_blob_settings(),
        )
        assert blob_store is not None
        workspace_repository = PostgresPaperWorkspaceRepository(session_factory)
        blob_repository = PostgresBlobArtifactRepository(session_factory)
        handler = PersistingMinioPdfHandler(
            session_store=PostgresSessionStore(session_factory),
            workspace_repository=workspace_repository,
        )
        service = PaperIntelService(
            handler=handler,
            artifact_repository=workspace_repository,
            blob_store=blob_store,
            blob_artifact_repository=blob_repository,
        )
        session = service.create_session()
        source_path = tmp_path / "minio-paper.pdf"
        source_path.write_bytes(PDF_BYTES)

        service.analyze_pdf(session.id, str(source_path), paper_id="local-minio-paper")
        artifact = blob_repository.get_by_kind_and_hash("pdf", expected_hash)
        assert artifact is not None
        with session_factory() as db:
            assert db.execute(select(func.count(BlobArtifactORM.id))).scalar_one() == 1
        assert blob_store.exists(artifact.object_key)
        assert [reference.ref_kind for reference in blob_repository.list_references(artifact.id)] == [
            "session",
            "paper_workspace",
        ]
        assert all(not Path(path).exists() for path in handler.materialized_paths)
    finally:
        try:
            if blob_store is not None and blob_store.exists(object_key):
                blob_store.delete(object_key)
        finally:
            with session_factory() as db:
                clear_foundation_tables(db)
            engine.dispose()


def test_real_minio_cleanup_deletes_staging_and_unref_ttl_objects():
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for MinIO cleanup test")
    if not _minio_url():
        pytest.skip("PAPERINTEL_MINIO_TEST_URL is required for MinIO cleanup test")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    blob_store = _store()
    blob_store.ensure_bucket()
    now = datetime.now(timezone.utc)
    staging_key = "uploads/minio-cleanup/expired.pdf"
    ttl_hash = hashlib.sha256(PNG_BYTES).hexdigest()
    ttl_key = f"page_images/sha256/{ttl_hash[:2]}/{ttl_hash}.png"

    try:
        with session_factory() as db:
            clear_foundation_tables(db)

        session = PostgresSessionStore(session_factory).create_session()
        upload_repository = PostgresPdfUploadRepository(session_factory)
        upload = upload_repository.create(
            PdfUpload(
                session_id=session.id,
                object_key=staging_key,
                expected_sha256=hashlib.sha256(PDF_BYTES).hexdigest(),
                expires_at=now - timedelta(minutes=5),
            )
        )
        blob_store.put_staging(
            staging_key,
            PDF_BYTES,
            content_type="application/pdf",
        )
        blob_repository = PostgresBlobArtifactRepository(session_factory)
        ttl_artifact = blob_repository.upsert_artifact(
            blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png"),
            retention_policy="ttl",
            expires_at=now - timedelta(hours=2),
        )

        summary = BlobCleanupService(
            repository=PostgresBlobCleanupRepository(session_factory),
            blob_store=blob_store,
            blob_grace_period_seconds=3600,
        ).run_once(now=now)

        assert summary.expired_uploads == 1
        assert summary.deleted_staging_objects == 1
        assert summary.released_blobs == 1
        assert summary.deleted_blob_objects == 1
        assert summary.errors == []
        assert upload_repository.get(upload.id).status == "expired"
        assert blob_repository.get_artifact(ttl_artifact.id) is None
        assert not blob_store.exists(staging_key)
        assert not blob_store.exists(ttl_key)
    finally:
        try:
            if blob_store.exists(staging_key):
                blob_store.delete(staging_key)
            if blob_store.exists(ttl_key):
                blob_store.delete(ttl_key)
        finally:
            with session_factory() as db:
                clear_foundation_tables(db)
            engine.dispose()
