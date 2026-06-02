from datetime import datetime, timedelta, timezone

import pytest

from models.blob_artifacts import BlobArtifact
from models.pdf_uploads import PdfUpload
from services.blob_cleanup import BlobCleanupService
from services.blob_store import BlobNotFoundError, BlobStoreUnavailableError


NOW = datetime(2026, 6, 2, tzinfo=timezone.utc)


class FakeCleanupRepository:
    def __init__(self, *, uploads=None, blobs=None) -> None:
        self.uploads = list(uploads or [])
        self.blobs = list(blobs or [])
        self.upload_dry_run_calls = []
        self.blob_dry_run_calls = []

    def list_expired_upload_candidates(self, *, now, limit):
        self.upload_dry_run_calls.append((now, limit))
        return self.uploads[:limit]

    def list_ttl_blob_cleanup_candidates(self, *, cutoff, limit):
        self.blob_dry_run_calls.append((cutoff, limit))
        return self.blobs[:limit]

    def expire_next_upload(self, *, now, delete_object):
        if not self.uploads:
            return None
        upload = self.uploads.pop(0)
        delete_object(upload.object_key)
        return upload

    def tombstone_next_ttl_blob(self, *, cutoff, now, delete_object):
        if not self.blobs:
            return None
        blob = self.blobs.pop(0)
        delete_object(blob.object_key)
        return blob


class FakeBlobStore:
    def __init__(self, *, error=None) -> None:
        self.error = error
        self.deleted = []

    def delete(self, object_key):
        self.deleted.append(object_key)
        if self.error is not None:
            raise self.error


def _upload(upload_id):
    return PdfUpload(
        id=upload_id,
        session_id="session-1",
        object_key=f"uploads/session-1/{upload_id}.pdf",
        expires_at=NOW - timedelta(seconds=1),
    )


def _ttl_blob(blob_id):
    return BlobArtifact(
        id=blob_id,
        kind="page_image",
        object_key=f"page_images/sha256/aa/{blob_id}.png",
        bucket_name="paperintel",
        content_hash="a" * 64,
        content_type="image/png",
        size_bytes=128,
        retention_policy="ttl",
        expires_at=NOW - timedelta(hours=2),
    )


def test_cleanup_service_expires_uploads_and_tombstones_ttl_blobs():
    repository = FakeCleanupRepository(
        uploads=[_upload("upload-1")], blobs=[_ttl_blob("blob-1")]
    )
    blob_store = FakeBlobStore()

    summary = BlobCleanupService(
        repository=repository,
        blob_store=blob_store,
        blob_grace_period_seconds=3600,
    ).run_once(now=NOW)

    assert summary.expired_uploads == 1
    assert summary.deleted_staging_objects == 1
    assert summary.released_blobs == 1
    assert summary.deleted_blob_objects == 1
    assert summary.errors == []
    assert blob_store.deleted == [
        "uploads/session-1/upload-1.pdf",
        "page_images/sha256/aa/blob-1.png",
    ]


def test_cleanup_service_dry_run_is_read_only_and_bounded():
    repository = FakeCleanupRepository(
        uploads=[_upload("upload-1"), _upload("upload-2")],
        blobs=[_ttl_blob("blob-1"), _ttl_blob("blob-2")],
    )
    blob_store = FakeBlobStore()

    summary = BlobCleanupService(
        repository=repository,
        blob_store=blob_store,
        upload_expiry_batch_size=1,
        blob_batch_size=1,
        blob_grace_period_seconds=60,
    ).run_once(dry_run=True, now=NOW)

    assert summary.skipped == 2
    assert repository.upload_dry_run_calls == [(NOW, 1)]
    assert repository.blob_dry_run_calls == [(NOW - timedelta(seconds=60), 1)]
    assert blob_store.deleted == []


def test_cleanup_service_treats_missing_object_as_success():
    repository = FakeCleanupRepository(uploads=[_upload("upload-1")])
    blob_store = FakeBlobStore(error=BlobNotFoundError("missing"))

    summary = BlobCleanupService(
        repository=repository, blob_store=blob_store
    ).run_once(now=NOW)

    assert summary.expired_uploads == 1
    assert summary.errors == []


def test_cleanup_service_stops_batch_on_storage_outage():
    repository = FakeCleanupRepository(uploads=[_upload("upload-1"), _upload("upload-2")])
    blob_store = FakeBlobStore(error=BlobStoreUnavailableError("storage down"))

    summary = BlobCleanupService(
        repository=repository, blob_store=blob_store
    ).run_once(now=NOW)

    assert summary.expired_uploads == 0
    assert summary.errors == [
        {"code": "staging_delete_failed", "message": "storage down"}
    ]
    assert blob_store.deleted == ["uploads/session-1/upload-1.pdf"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upload_expiry_batch_size": 0},
        {"blob_batch_size": 0},
        {"blob_grace_period_seconds": -1},
    ],
)
def test_cleanup_service_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BlobCleanupService(
            repository=FakeCleanupRepository(),
            blob_store=FakeBlobStore(),
            **kwargs,
        )
