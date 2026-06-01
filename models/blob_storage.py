from typing import Literal, TypeAlias

from pydantic import BaseModel, Field


BlobKind: TypeAlias = Literal["pdf", "page_image", "generated_artifact"]


class StoredBlobObject(BaseModel):
    """Physical object descriptor returned by a BlobStore."""

    kind: BlobKind
    object_key: str
    bucket_name: str
    content_hash: str = Field(min_length=64, max_length=64)
    content_type: str
    size_bytes: int = Field(ge=0)
    storage_backend: str = "s3"
