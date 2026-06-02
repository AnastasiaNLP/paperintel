from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.pdf_uploads import PdfUpload


def test_pdf_upload_accepts_initiated_record_without_blob():
    upload = PdfUpload(
        session_id="session-1",
        object_key="uploads/session-1/upload-1.pdf",
        expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )

    assert upload.status == "initiated"
    assert upload.blob_id is None


def test_pdf_upload_finalized_requires_blob_hash_and_timestamp():
    with pytest.raises(ValidationError, match="finalized uploads require blob_id"):
        PdfUpload(
            session_id="session-1",
            object_key="uploads/session-1/upload-1.pdf",
            status="finalized",
            expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        )


def test_pdf_upload_finalized_requires_expected_sha256():
    with pytest.raises(ValidationError, match="both SHA-256 values"):
        PdfUpload(
            session_id="session-1",
            blob_id="blob-1",
            object_key="uploads/session-1/upload-1.pdf",
            actual_sha256="a" * 64,
            status="finalized",
            finalized_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
            expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        )


def test_pdf_upload_finalized_requires_matching_sha256():
    with pytest.raises(ValidationError, match="SHA-256 values must match"):
        PdfUpload(
            session_id="session-1",
            blob_id="blob-1",
            object_key="uploads/session-1/upload-1.pdf",
            expected_sha256="a" * 64,
            actual_sha256="b" * 64,
            status="finalized",
            finalized_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
            expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        )


def test_pdf_upload_rejects_invalid_sha256():
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        PdfUpload(
            session_id="session-1",
            object_key="uploads/session-1/upload-1.pdf",
            expected_sha256="Z" * 64,
            expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        )
