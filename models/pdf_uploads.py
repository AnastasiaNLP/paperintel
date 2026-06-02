from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from models.session import utc_now


PdfUploadStatus: TypeAlias = Literal[
    "initiated",
    "uploaded",
    "finalized",
    "enqueued",
    "failed",
    "expired",
]


class PdfUpload(BaseModel):
    """Durable lifecycle record for one client PDF upload."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    blob_id: str | None = None
    object_key: str
    expected_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    actual_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    size_bytes: int | None = Field(default=None, ge=0)
    content_type: str = "application/pdf"
    status: PdfUploadStatus = "initiated"
    expires_at: datetime
    finalized_at: datetime | None = None
    error_json: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("session_id", "object_key", "content_type")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_sha256", "actual_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and any(character not in "0123456789abcdef" for character in value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_finalized_state(self) -> "PdfUpload":
        if self.status in {"finalized", "enqueued"}:
            if self.blob_id is None:
                raise ValueError("finalized uploads require blob_id")
            if self.expected_sha256 is None or self.actual_sha256 is None:
                raise ValueError("finalized uploads require both SHA-256 values")
            if self.expected_sha256 != self.actual_sha256:
                raise ValueError("finalized upload SHA-256 values must match")
            if self.finalized_at is None:
                raise ValueError("finalized uploads require finalized_at")
        return self
