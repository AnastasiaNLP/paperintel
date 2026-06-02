import hashlib
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from services.blob_store import (
    BlobConfigurationError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobSizeLimitError,
    BlobStoreUnavailableError,
    S3BlobStore,
    object_key_for,
)


BUCKET = "paperintel-test-blobs"
PDF_BYTES = b"%PDF-1.7\npaperintel blob test\n"


def _client():
    return boto3.client("s3", region_name="us-east-1")


@mock_aws
def test_ensure_bucket_creates_missing_bucket_and_is_idempotent():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)

    store.ensure_bucket()
    store.ensure_bucket()

    names = [bucket["Name"] for bucket in client.list_buckets()["Buckets"]]
    assert names == [BUCKET]


@mock_aws
def test_ensure_bucket_supports_non_default_aws_region():
    client = boto3.client("s3", region_name="eu-west-1")
    store = S3BlobStore(client=client, bucket_name=BUCKET)

    store.ensure_bucket()

    names = [bucket["Name"] for bucket in client.list_buckets()["Buckets"]]
    assert names == [BUCKET]


@mock_aws
def test_put_pdf_uses_content_addressed_key_and_real_s3_metadata():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()

    stored = store.put(PDF_BYTES, kind="pdf")

    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    assert stored.content_hash == expected_hash
    assert stored.object_key == f"papers/sha256/{expected_hash[:2]}/{expected_hash}.pdf"
    assert stored.content_type == "application/pdf"
    assert stored.size_bytes == len(PDF_BYTES)
    response = client.head_object(Bucket=BUCKET, Key=stored.object_key)
    assert response["ContentType"] == "application/pdf"
    assert response["ContentLength"] == len(PDF_BYTES)


@mock_aws
def test_put_is_idempotent_and_does_not_upload_existing_object(monkeypatch):
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    put_calls = []
    real_put_object = client.put_object

    def tracked_put_object(**kwargs):
        put_calls.append(kwargs)
        return real_put_object(**kwargs)

    monkeypatch.setattr(client, "put_object", tracked_put_object)

    first = store.put(PDF_BYTES, kind="pdf")
    second = store.put(PDF_BYTES, kind="pdf")

    assert first == second
    assert len(put_calls) == 1


@mock_aws
def test_idempotent_put_returns_persisted_s3_content_type():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    stored = store.put(b"json-content", kind="generated_artifact")
    client.copy_object(
        Bucket=BUCKET,
        Key=stored.object_key,
        CopySource={"Bucket": BUCKET, "Key": stored.object_key},
        ContentType="application/json",
        MetadataDirective="REPLACE",
    )

    reused = store.put(b"json-content", kind="generated_artifact")

    assert reused.content_type == "application/json"
    assert reused.size_bytes == len(b"json-content")


@mock_aws
def test_put_supports_generic_blob_kinds_with_canonical_content_type():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()

    stored = store.put(b"png-content", kind="page_image")

    assert stored.object_key.startswith("page_images/sha256/")
    assert stored.object_key.endswith(".png")
    assert stored.content_type == "image/png"


def test_put_rejects_content_type_that_does_not_match_kind():
    store = S3BlobStore(client=object(), bucket_name=BUCKET)

    with pytest.raises(BlobConfigurationError):
        store.put(b"webp-content", kind="page_image", content_type="image/webp")


@mock_aws
def test_materialize_downloads_real_object_and_deletes_temporary_file():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    stored = store.put(PDF_BYTES, kind="pdf")

    with store.materialize(
        stored.object_key,
        expected_sha256=stored.content_hash,
    ) as pdf_path:
        materialized = Path(pdf_path)
        assert materialized.exists()
        assert materialized.read_bytes() == PDF_BYTES

    assert not materialized.exists()


@mock_aws
def test_materialize_deletes_temporary_file_when_caller_raises():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    stored = store.put(PDF_BYTES, kind="pdf")
    materialized = None

    with pytest.raises(RuntimeError):
        with store.materialize(stored.object_key) as pdf_path:
            materialized = Path(pdf_path)
            assert materialized.exists()
            raise RuntimeError("caller failed")

    assert materialized is not None
    assert not materialized.exists()


@mock_aws
def test_materialize_rejects_integrity_mismatch_without_leaking_temp_file():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    stored = store.put(PDF_BYTES, kind="pdf")

    with pytest.raises(BlobIntegrityError):
        with store.materialize(stored.object_key, expected_sha256="0" * 64):
            pytest.fail("materialize must fail before yielding a path")


@mock_aws
def test_delete_removes_object_and_missing_materialize_is_explicit():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    stored = store.put(PDF_BYTES, kind="pdf")

    store.delete(stored.object_key)

    assert not store.exists(stored.object_key)
    with pytest.raises(BlobNotFoundError):
        with store.materialize(stored.object_key):
            pytest.fail("missing blob must not yield a path")


def test_empty_bucket_name_is_configuration_error():
    with pytest.raises(BlobConfigurationError):
        S3BlobStore(client=object(), bucket_name="  ")


def test_object_key_requires_sha256_hex_digest():
    with pytest.raises(ValueError):
        object_key_for("pdf", "too-short")
    with pytest.raises(ValueError):
        object_key_for("pdf", "z" * 64)


def test_exists_wraps_provider_failure_as_unavailable():
    class UnavailableClient:
        def head_object(self, **kwargs):
            raise RuntimeError("provider down")

    store = S3BlobStore(client=UnavailableClient(), bucket_name=BUCKET)

    with pytest.raises(BlobStoreUnavailableError):
        store.exists("papers/sha256/00/example.pdf")


def test_ensure_bucket_wraps_permission_failure_as_configuration_error():
    class ForbiddenClient:
        def head_bucket(self, **kwargs):
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": "denied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "HeadBucket",
            )

    store = S3BlobStore(client=ForbiddenClient(), bucket_name=BUCKET)

    with pytest.raises(BlobConfigurationError):
        store.ensure_bucket()


def test_ensure_bucket_wraps_runtime_outage_as_unavailable():
    class UnavailableClient:
        def head_bucket(self, **kwargs):
            raise RuntimeError("connection refused")

    store = S3BlobStore(client=UnavailableClient(), bucket_name=BUCKET)

    with pytest.raises(BlobStoreUnavailableError):
        store.ensure_bucket()


@mock_aws
def test_staging_upload_and_presigned_put_use_explicit_object_key():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    object_key = "uploads/session-1/upload-1.pdf"

    upload_url = store.create_presigned_put(
        object_key, content_type="application/pdf", expires_seconds=900
    )
    store.put_staging(object_key, PDF_BYTES, content_type="application/pdf")

    assert object_key in upload_url
    assert store.exists(object_key)
    with store.materialize(object_key) as path:
        assert Path(path).read_bytes() == PDF_BYTES
    store.delete(object_key)
    assert not store.exists(object_key)


@mock_aws
def test_head_object_returns_metadata_without_downloading_body():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    object_key = "uploads/session-1/upload-1.pdf"
    store.put_staging(object_key, PDF_BYTES, content_type="application/pdf")

    metadata = store.head_object(object_key)

    assert metadata.object_key == object_key
    assert metadata.content_type == "application/pdf"
    assert metadata.size_bytes == len(PDF_BYTES)


@mock_aws
def test_head_object_rejects_missing_object():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()

    with pytest.raises(BlobNotFoundError):
        store.head_object("uploads/session-1/missing.pdf")


@mock_aws
def test_materialize_enforces_download_limit_even_after_head_check():
    client = _client()
    store = S3BlobStore(client=client, bucket_name=BUCKET)
    store.ensure_bucket()
    object_key = "uploads/session-1/replaced.pdf"
    store.put_staging(object_key, b"too-large", content_type="application/pdf")

    with pytest.raises(BlobSizeLimitError):
        with store.materialize(object_key, max_bytes=3):
            pytest.fail("oversized object must not yield a path")


def test_materialize_closes_response_body_after_successful_read():
    body = TrackingBody(PDF_BYTES)
    store = S3BlobStore(client=BodyClient(body), bucket_name=BUCKET)

    with store.materialize("papers/example.pdf") as path:
        assert Path(path).read_bytes() == PDF_BYTES

    assert body.closed is True


def test_materialize_closes_response_body_after_limited_read():
    body = TrackingBody(b"too-large")
    store = S3BlobStore(client=BodyClient(body), bucket_name=BUCKET)

    with pytest.raises(BlobSizeLimitError):
        with store.materialize("uploads/session-1/replaced.pdf", max_bytes=3):
            pytest.fail("oversized object must not yield a path")

    assert body.closed is True


class TrackingBody:
    def __init__(self, content):
        self.content = content
        self.closed = False

    def read(self, amount=None):
        return self.content if amount is None else self.content[:amount]

    def close(self):
        self.closed = True


class BodyClient:
    def __init__(self, body):
        self.body = body

    def get_object(self, **kwargs):
        return {"Body": self.body}
