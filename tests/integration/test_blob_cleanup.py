import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import boto3
import pytest
from alembic import command
from alembic.config import Config
from moto import mock_aws

from models.pdf_uploads import PdfUpload
from services.blob_cleanup import BlobCleanupService
from services.blob_store import BlobStoreUnavailableError, S3BlobStore
from storage.db import make_engine, make_session_factory
from storage.models import BlobArtifactORM
from storage.repositories import (
    PostgresBlobArtifactRepository,
    PostgresBlobCleanupRepository,
    PostgresPdfUploadRepository,
    PostgresSessionStore,
    clear_foundation_tables,
)


pytestmark = pytest.mark.db
BUCKET = "paperintel-cleanup-test"
PDF_BYTES = b"%PDF-1.7\ncleanup\n"
PNG_BYTES = b"png-cleanup"


@pytest.fixture()
def stack():
    database_url = os.environ.get("PAPERINTEL_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for cleanup tests")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    with session_factory() as db:
        clear_foundation_tables(db)
    yield session_factory
    with session_factory() as db:
        clear_foundation_tables(db)
    engine.dispose()


def _blob_store():
    client = boto3.client("s3", region_name="us-east-1")
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    return client, store


def _expired_upload(repository, session_id, upload_id, *, status="initiated"):
    upload = repository.create(
        PdfUpload(
            id=upload_id,
            session_id=session_id,
            object_key=f"uploads/{session_id}/{upload_id}.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )
    if status == "uploaded":
        return repository.mark_uploaded(upload.id)
    if status == "failed":
        return repository.mark_failed(upload.id, error_json={"code": "original"})
    return upload


def _active_upload(repository, session_id, upload_id):
    return repository.create(
        PdfUpload(
            id=upload_id,
            session_id=session_id,
            object_key=f"uploads/{session_id}/{upload_id}.pdf",
            expected_sha256="a" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )


@mock_aws
def test_cleanup_expires_staging_uploads_but_skips_finalized_upload(stack):
    client, blob_store = _blob_store()
    session = PostgresSessionStore(stack).create_session()
    upload_repository = PostgresPdfUploadRepository(stack)
    for status in ("initiated", "uploaded", "failed"):
        upload = _expired_upload(upload_repository, session.id, status, status=status)
        blob_store.put_staging(upload.object_key, PDF_BYTES, content_type="application/pdf")
    artifact_repository = PostgresBlobArtifactRepository(stack)
    stored = blob_store.put(PDF_BYTES, kind="pdf", content_type="application/pdf")
    artifact = artifact_repository.upsert_artifact(stored)
    finalized = upload_repository.mark_uploaded(
        _active_upload(upload_repository, session.id, "finalized").id
    )
    upload_repository.finalize(
        finalized.id,
        blob_id=artifact.id,
        actual_sha256="a" * 64,
        size_bytes=len(PDF_BYTES),
    )
    enqueued = upload_repository.mark_uploaded(
        _active_upload(upload_repository, session.id, "enqueued").id
    )
    upload_repository.finalize(
        enqueued.id,
        blob_id=artifact.id,
        actual_sha256="a" * 64,
        size_bytes=len(PDF_BYTES),
    )
    upload_repository.mark_enqueued(enqueued.id)

    summary = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack), blob_store=blob_store
    ).run_once(now=datetime.now(timezone.utc))

    assert summary.expired_uploads == 3
    assert summary.deleted_staging_objects == 3
    assert upload_repository.get("initiated").status == "expired"
    assert upload_repository.get("uploaded").status == "expired"
    failed = upload_repository.get("failed")
    assert failed.status == "expired"
    assert failed.error_json["previous_error"] == {"code": "original"}
    assert upload_repository.get("finalized").status == "finalized"
    assert upload_repository.get("enqueued").status == "enqueued"
    keys = {item["Key"] for item in client.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert keys == {artifact.object_key}


@mock_aws
def test_cleanup_tombstones_unreferenced_ttl_blob_and_preserves_history(stack):
    client, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    stored = blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png")
    artifact = repository.upsert_artifact(
        stored,
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    repository.add_reference(artifact.id, ref_kind="session", ref_id="session-1")
    repository.release_reference(artifact.id, ref_kind="session", ref_id="session-1")

    summary = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=blob_store,
        blob_grace_period_seconds=3600,
    ).run_once(now=datetime.now(timezone.utc))

    assert summary.released_blobs == 1
    assert summary.deleted_blob_objects == 1
    assert repository.get_artifact(artifact.id) is None
    assert repository.list_references(artifact.id)[0].status == "released"
    with stack() as db:
        tombstone = db.get(BlobArtifactORM, artifact.id)
        assert tombstone.status == "deleted"
        assert tombstone.deleted_at is not None
        assert tombstone.cleanup_metadata_json["code"] == "ttl_blob_deleted"
    assert client.list_objects_v2(Bucket=BUCKET).get("Contents", []) == []


@mock_aws
def test_cleanup_keeps_active_reference_and_dry_run_is_read_only(stack):
    client, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    stored = blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png")
    artifact = repository.upsert_artifact(
        stored,
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    repository.add_reference(artifact.id, ref_kind="session", ref_id="session-1")
    cleanup = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=blob_store,
        blob_grace_period_seconds=0,
    )

    held = cleanup.run_once(now=datetime.now(timezone.utc))
    repository.release_reference(artifact.id, ref_kind="session", ref_id="session-1")
    dry_run = cleanup.run_once(dry_run=True, now=datetime.now(timezone.utc))

    assert held.released_blobs == 0
    assert dry_run.skipped == 1
    assert repository.get_artifact(artifact.id) is not None
    assert len(client.list_objects_v2(Bucket=BUCKET)["Contents"]) == 1


@mock_aws
def test_cleanup_storage_outage_leaves_upload_retryable(stack):
    _, blob_store = _blob_store()
    session = PostgresSessionStore(stack).create_session()
    upload_repository = PostgresPdfUploadRepository(stack)
    upload = _expired_upload(upload_repository, session.id, "upload-1")

    class UnavailableStore:
        def delete(self, object_key):
            raise BlobStoreUnavailableError("storage down")

    summary = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=UnavailableStore(),
    ).run_once(now=datetime.now(timezone.utc))

    assert summary.errors[0]["code"] == "staging_delete_failed"
    assert upload_repository.get(upload.id).status == "initiated"


@mock_aws
def test_cleanup_storage_outage_leaves_ttl_blob_active(stack):
    _, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    artifact = repository.upsert_artifact(
        blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png"),
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )

    class UnavailableStore:
        def delete(self, object_key):
            raise BlobStoreUnavailableError("storage down")

    summary = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=UnavailableStore(),
        blob_grace_period_seconds=0,
    ).run_once(now=datetime.now(timezone.utc))

    assert summary.errors[0]["code"] == "blob_delete_failed"
    assert repository.get_artifact(artifact.id) is not None


@mock_aws
def test_cleanup_tombstone_is_reactivated_by_canonical_upsert(stack):
    _, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    stored = blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png")
    artifact = repository.upsert_artifact(
        stored,
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    repository.add_reference(artifact.id, ref_kind="session", ref_id="session-1")
    repository.release_reference(artifact.id, ref_kind="session", ref_id="session-1")
    BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=blob_store,
        blob_grace_period_seconds=0,
    ).run_once(now=datetime.now(timezone.utc))

    restored_object = blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png")
    restored = repository.upsert_artifact(
        restored_object,
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    restored_reference = repository.add_reference(
        restored.id, ref_kind="session", ref_id="session-1"
    )

    assert restored.id == artifact.id
    assert restored.status == "active"
    assert restored.deleted_at is None
    assert restored.cleanup_metadata == {}
    assert restored_reference.status == "active"
    assert restored_reference.released_at is None


@mock_aws
def test_cleanup_tombstone_rejects_new_reference_and_durable_blob_is_protected(stack):
    _, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    ttl = repository.upsert_artifact(
        blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png"),
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    durable = repository.upsert_artifact(
        blob_store.put(PDF_BYTES, kind="pdf", content_type="application/pdf")
    )
    BlobCleanupService(
        repository=PostgresBlobCleanupRepository(stack),
        blob_store=blob_store,
        blob_grace_period_seconds=0,
    ).run_once(now=datetime.now(timezone.utc))

    with pytest.raises(ValueError, match="Blob artifact not found"):
        repository.add_reference(ttl.id, ref_kind="session", ref_id="session-1")
    assert repository.get_artifact(durable.id) is not None


def test_cleanup_repository_respects_upload_batch_limit(stack):
    session = PostgresSessionStore(stack).create_session()
    upload_repository = PostgresPdfUploadRepository(stack)
    _expired_upload(upload_repository, session.id, "upload-1")
    _expired_upload(upload_repository, session.id, "upload-2")

    candidates = PostgresBlobCleanupRepository(stack).list_expired_upload_candidates(
        now=datetime.now(timezone.utc), limit=1
    )

    assert len(candidates) == 1


def test_cleanup_workers_skip_locked_expired_upload(stack):
    session = PostgresSessionStore(stack).create_session()
    upload_repository = PostgresPdfUploadRepository(stack)
    upload = _expired_upload(upload_repository, session.id, "upload-1")
    delete_started = Event()
    release_delete = Event()

    def blocking_delete(object_key):
        delete_started.set()
        assert release_delete.wait(timeout=5)

    def first_cleanup():
        return PostgresBlobCleanupRepository(stack).expire_next_upload(
            now=datetime.now(timezone.utc), delete_object=blocking_delete
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_cleanup)
        assert delete_started.wait(timeout=5)
        second = PostgresBlobCleanupRepository(stack).expire_next_upload(
            now=datetime.now(timezone.utc), delete_object=lambda _: None
        )
        release_delete.set()
        cleaned = first.result(timeout=5)

    assert second is None
    assert cleaned.id == upload.id


@mock_aws
def test_cleanup_workers_skip_locked_ttl_blob(stack):
    _, blob_store = _blob_store()
    repository = PostgresBlobArtifactRepository(stack)
    artifact = repository.upsert_artifact(
        blob_store.put(PNG_BYTES, kind="page_image", content_type="image/png"),
        retention_policy="ttl",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    delete_started = Event()
    release_delete = Event()

    def blocking_delete(object_key):
        delete_started.set()
        assert release_delete.wait(timeout=5)

    def first_cleanup():
        return PostgresBlobCleanupRepository(stack).tombstone_next_ttl_blob(
            cutoff=datetime.now(timezone.utc) - timedelta(hours=1),
            now=datetime.now(timezone.utc),
            delete_object=blocking_delete,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_cleanup)
        assert delete_started.wait(timeout=5)
        second = PostgresBlobCleanupRepository(stack).tombstone_next_ttl_blob(
            cutoff=datetime.now(timezone.utc) - timedelta(hours=1),
            now=datetime.now(timezone.utc),
            delete_object=lambda _: None,
        )
        release_delete.set()
        cleaned = first.result(timeout=5)

    assert second is None
    assert cleaned.id == artifact.id
    assert repository.get_artifact(artifact.id) is None
