from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from models.session import utc_now
from services.blob_store import BlobNotFoundError, BlobStore, BlobStoreUnavailableError
from storage.repositories import PostgresBlobCleanupRepository


class BlobCleanupSummary(BaseModel):
    expired_uploads: int = 0
    deleted_staging_objects: int = 0
    released_blobs: int = 0
    deleted_blob_objects: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)


class BlobCleanupService:
    def __init__(
        self,
        *,
        repository: PostgresBlobCleanupRepository,
        blob_store: BlobStore,
        upload_expiry_batch_size: int = 100,
        blob_batch_size: int = 100,
        blob_grace_period_seconds: int = 3600,
    ) -> None:
        if upload_expiry_batch_size <= 0:
            raise ValueError("upload_expiry_batch_size must be positive.")
        if blob_batch_size <= 0:
            raise ValueError("blob_batch_size must be positive.")
        if blob_grace_period_seconds < 0:
            raise ValueError("blob_grace_period_seconds must not be negative.")
        self.repository = repository
        self.blob_store = blob_store
        self.upload_expiry_batch_size = upload_expiry_batch_size
        self.blob_batch_size = blob_batch_size
        self.blob_grace_period_seconds = blob_grace_period_seconds

    def run_once(
        self, *, dry_run: bool = False, now: datetime | None = None
    ) -> BlobCleanupSummary:
        now = now or utc_now()
        summary = BlobCleanupSummary()
        if dry_run:
            uploads = self.repository.list_expired_upload_candidates(
                now=now, limit=self.upload_expiry_batch_size
            )
            blobs = self.repository.list_ttl_blob_cleanup_candidates(
                cutoff=self._blob_cutoff(now), limit=self.blob_batch_size
            )
            summary.skipped = len(uploads) + len(blobs)
            return summary

        for _ in range(self.upload_expiry_batch_size):
            try:
                upload = self.repository.expire_next_upload(
                    now=now, delete_object=self._delete_object
                )
            except BlobStoreUnavailableError as exc:
                summary.errors.append(_cleanup_error("staging_delete_failed", exc))
                break
            if upload is None:
                break
            summary.expired_uploads += 1
            summary.deleted_staging_objects += 1

        for _ in range(self.blob_batch_size):
            try:
                blob = self.repository.tombstone_next_ttl_blob(
                    cutoff=self._blob_cutoff(now),
                    now=now,
                    delete_object=self._delete_object,
                )
            except BlobStoreUnavailableError as exc:
                summary.errors.append(_cleanup_error("blob_delete_failed", exc))
                break
            if blob is None:
                break
            summary.released_blobs += 1
            summary.deleted_blob_objects += 1
        return summary

    def _blob_cutoff(self, now: datetime) -> datetime:
        return now - timedelta(seconds=self.blob_grace_period_seconds)

    def _delete_object(self, object_key: str) -> None:
        try:
            self.blob_store.delete(object_key)
        except BlobNotFoundError:
            return


def _cleanup_error(code: str, exc: Exception) -> dict[str, str]:
    return {"code": code, "message": str(exc)}
