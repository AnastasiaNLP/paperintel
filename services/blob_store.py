from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ContextManager, Iterator, Protocol

from models.blob_storage import BlobKind, StoredBlobObject


DEFAULT_CONTENT_TYPES: dict[BlobKind, str] = {
    "pdf": "application/pdf",
    "page_image": "image/png",
    "generated_artifact": "application/octet-stream",
}
OBJECT_PREFIXES: dict[BlobKind, str] = {
    "pdf": "papers",
    "page_image": "page_images",
    "generated_artifact": "generated_artifacts",
}
OBJECT_SUFFIXES: dict[BlobKind, str] = {
    "pdf": ".pdf",
    "page_image": ".png",
    "generated_artifact": ".bin",
}


class BlobStoreError(RuntimeError):
    pass


class BlobNotFoundError(BlobStoreError):
    pass


class BlobStoreUnavailableError(BlobStoreError):
    pass


class BlobIntegrityError(BlobStoreError):
    pass


class BlobConfigurationError(BlobStoreError):
    pass


class BlobStore(Protocol):
    def ensure_bucket(self) -> None: ...

    def put(
        self,
        content: bytes,
        *,
        kind: BlobKind,
        content_type: str | None = None,
    ) -> StoredBlobObject: ...

    def exists(self, object_key: str) -> bool: ...

    def materialize(
        self,
        object_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> ContextManager[str]: ...

    def delete(self, object_key: str) -> None: ...


class S3BlobStore:
    """Synchronous S3-compatible adapter for durable binary objects."""

    def __init__(self, *, client: Any, bucket_name: str) -> None:
        bucket_name = bucket_name.strip()
        if not bucket_name:
            raise BlobConfigurationError("Blob store bucket_name must not be empty.")
        self.client = client
        self.bucket_name = bucket_name

    @classmethod
    def from_config(
        cls,
        *,
        bucket_name: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> "S3BlobStore":
        try:
            import boto3
        except ImportError as exc:
            raise BlobConfigurationError(
                "boto3 is required for S3BlobStore.from_config()."
            ) from exc

        try:
            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region_name or None,
                aws_access_key_id=access_key_id or None,
                aws_secret_access_key=secret_access_key or None,
            )
        except Exception as exc:
            raise BlobConfigurationError(
                f"Could not configure S3 blob store client: {exc}"
            ) from exc
        return cls(client=client, bucket_name=bucket_name)

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return
        except Exception as exc:
            if not _is_missing_error(exc):
                raise _bucket_error("inspect", self.bucket_name, exc) from exc

        try:
            self.client.create_bucket(**self._create_bucket_kwargs())
        except Exception as exc:
            raise _bucket_error("create", self.bucket_name, exc) from exc

    def put(
        self,
        content: bytes,
        *,
        kind: BlobKind,
        content_type: str | None = None,
    ) -> StoredBlobObject:
        if not isinstance(content, bytes):
            raise TypeError("BlobStore.put content must be bytes.")
        resolved_content_type = content_type or DEFAULT_CONTENT_TYPES[kind]
        _validate_content_type(kind, resolved_content_type)
        content_hash = hashlib.sha256(content).hexdigest()
        object_key = object_key_for(kind, content_hash)
        existing = self._head_object_or_none(object_key)
        if existing is None:
            try:
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=content,
                    ContentType=resolved_content_type,
                )
            except Exception as exc:
                raise BlobStoreUnavailableError(
                    f"Could not upload blob object {object_key!r}: {exc}"
                ) from exc
            stored_content_type = resolved_content_type
            stored_size = len(content)
        else:
            stored_content_type = str(existing.get("ContentType") or resolved_content_type)
            stored_size = int(existing.get("ContentLength", len(content)))
        return StoredBlobObject(
            kind=kind,
            object_key=object_key,
            bucket_name=self.bucket_name,
            content_hash=content_hash,
            content_type=stored_content_type,
            size_bytes=stored_size,
        )

    def exists(self, object_key: str) -> bool:
        return self._head_object_or_none(object_key) is not None

    def _head_object_or_none(self, object_key: str) -> dict[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.bucket_name, Key=object_key)
        except Exception as exc:
            if _is_missing_error(exc):
                return None
            raise BlobStoreUnavailableError(
                f"Could not inspect blob object {object_key!r}: {exc}"
            ) from exc

    @contextmanager
    def materialize(
        self,
        object_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> Iterator[str]:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
            content = response["Body"].read()
        except Exception as exc:
            if _is_missing_error(exc):
                raise BlobNotFoundError(f"Blob object not found: {object_key}") from exc
            raise BlobStoreUnavailableError(
                f"Could not download blob object {object_key!r}: {exc}"
            ) from exc

        actual_sha256 = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise BlobIntegrityError(
                f"Blob integrity check failed for {object_key!r}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

        suffix = Path(object_key).suffix
        temp_path = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                prefix="paperintel_blob_",
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            yield temp_path
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

    def delete(self, object_key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
        except Exception as exc:
            raise BlobStoreUnavailableError(
                f"Could not delete blob object {object_key!r}: {exc}"
            ) from exc

    def _create_bucket_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"Bucket": self.bucket_name}
        region_name = getattr(getattr(self.client, "meta", None), "region_name", None)
        if region_name and region_name != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region_name,
            }
        return kwargs


def object_key_for(kind: BlobKind, content_hash: str) -> str:
    if len(content_hash) != 64:
        raise ValueError("content_hash must be a 64-character SHA-256 hex digest.")
    try:
        int(content_hash, 16)
    except ValueError as exc:
        raise ValueError("content_hash must be a hexadecimal SHA-256 digest.") from exc
    return (
        f"{OBJECT_PREFIXES[kind]}/sha256/{content_hash[:2]}/"
        f"{content_hash}{OBJECT_SUFFIXES[kind]}"
    )


def _is_missing_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"} or status == 404


def _validate_content_type(kind: BlobKind, content_type: str) -> None:
    expected = DEFAULT_CONTENT_TYPES[kind]
    if content_type != expected:
        raise BlobConfigurationError(
            f"Blob kind {kind!r} requires content type {expected!r}, got {content_type!r}."
        )


def _bucket_error(action: str, bucket_name: str, exc: Exception) -> BlobStoreError:
    if _is_provider_unavailable_error(exc):
        return BlobStoreUnavailableError(
            f"Could not {action} blob bucket {bucket_name!r}: {exc}"
        )
    return BlobConfigurationError(
        f"Could not {action} blob bucket {bucket_name!r}: {exc}"
    )


def _is_provider_unavailable_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return True
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return isinstance(status, int) and status >= 500
