from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from models.blob_artifacts import BlobArtifact, BlobReference


HASH = "a" * 64


def test_blob_artifact_accepts_durable_pdf_registry_entry():
    artifact = BlobArtifact(
        kind="pdf",
        object_key=f"papers/sha256/aa/{HASH}.pdf",
        bucket_name="paperintel",
        content_hash=HASH,
        content_type="application/pdf",
        size_bytes=128,
    )

    assert artifact.retention_policy == "durable"
    assert artifact.expires_at is None
    assert artifact.storage_backend == "s3"


def test_blob_artifact_ttl_requires_expiry():
    with pytest.raises(ValidationError, match="ttl blobs require expires_at"):
        BlobArtifact(
            kind="page_image",
            object_key=f"page_images/sha256/aa/{HASH}.png",
            bucket_name="paperintel",
            content_hash=HASH,
            content_type="image/png",
            size_bytes=128,
            retention_policy="ttl",
        )


def test_blob_artifact_durable_rejects_expiry():
    with pytest.raises(ValidationError, match="durable blobs must not have expires_at"):
        BlobArtifact(
            kind="pdf",
            object_key=f"papers/sha256/aa/{HASH}.pdf",
            bucket_name="paperintel",
            content_hash=HASH,
            content_type="application/pdf",
            size_bytes=128,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )


def test_blob_artifact_requires_lowercase_sha256_hex():
    with pytest.raises(ValidationError, match="lowercase SHA-256 hex digest"):
        BlobArtifact(
            kind="pdf",
            object_key="papers/sha256/zz/not-a-hash.pdf",
            bucket_name="paperintel",
            content_hash="Z" * 64,
            content_type="application/pdf",
            size_bytes=128,
        )


def test_blob_reference_accepts_polymorphic_relationship():
    reference = BlobReference(
        blob_id="blob-1",
        ref_kind="paper_workspace",
        ref_id="workspace-1",
        metadata={"source": "upload"},
    )

    assert reference.ref_kind == "paper_workspace"
    assert reference.metadata == {"source": "upload"}
