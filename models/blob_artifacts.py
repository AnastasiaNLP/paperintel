from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from models.blob_storage import BlobKind
from models.session import utc_now


BlobRetentionPolicy: TypeAlias = Literal["durable", "ttl"]
BlobReferenceKind: TypeAlias = Literal[
    "session",
    "paper_workspace",
    "workflow_job",
]


class BlobArtifact(BaseModel):
    """Durable registry entry for one physical object in blob storage."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: BlobKind
    object_key: str
    bucket_name: str
    content_hash: str = Field(min_length=64, max_length=64)
    content_type: str
    size_bytes: int = Field(ge=0)
    storage_backend: Literal["s3"] = "s3"
    retention_policy: BlobRetentionPolicy = "durable"
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("object_key", "bucket_name", "content_type")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_retention(self) -> "BlobArtifact":
        if self.retention_policy == "durable" and self.expires_at is not None:
            raise ValueError("durable blobs must not have expires_at")
        if self.retention_policy == "ttl" and self.expires_at is None:
            raise ValueError("ttl blobs require expires_at")
        return self


class BlobReference(BaseModel):
    """Usage relationship between a durable blob and an application entity."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    blob_id: str
    ref_kind: BlobReferenceKind
    ref_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("blob_id", "ref_id")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value
