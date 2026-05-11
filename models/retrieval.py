from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536

ChunkType = Literal[
    "text",
    "abstract",
    "table",
    "equation",
    "figure",
    "caption",
    "reference",
]
ArtifactType = Literal[
    "table",
    "equation",
    "figure",
    "page_image",
    "pdf",
    "raw_text",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChunkSource(BaseModel):
    """
    Stable paper identity and ingestion context for a retrievable chunk.

    For Stage C, paper_id is the canonical paper identifier, normally the arXiv
    id such as "2310.06825", not a session-scoped workspace UUID.
    """

    paper_id: str
    paper_index: int | None = None
    session_id: str | None = None
    input_url: str | None = None
    title: str | None = None
    arxiv_id: str | None = None

    @field_validator("paper_id")
    @classmethod
    def paper_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id must not be blank")
        return value


class ChunkLocation(BaseModel):
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class EvidenceArtifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    artifact_type: ArtifactType
    storage_ref: str | None = None
    label: str | None = None
    page: int | None = None
    bbox: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("paper_id")
    @classmethod
    def paper_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id must not be blank")
        return value


class PaperChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    chunk_index: int
    text: str
    chunk_type: ChunkType = "text"
    source: ChunkSource
    location: ChunkLocation = Field(default_factory=ChunkLocation)
    artifact_refs: list[EvidenceArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("paper_id")
    @classmethod
    def paper_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id must not be blank")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class ChunkSearchQuery(BaseModel):
    query: str
    session_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    limit: int = 5
    min_score: float | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("limit")
    @classmethod
    def limit_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("limit must be positive")
        return value


class ChunkSearchResult(BaseModel):
    chunk: PaperChunk
    score: float
    rank: int
    match_reason: str | None = None


class CitationRef(BaseModel):
    paper_id: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    artifact_refs: list[EvidenceArtifact] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    query: str
    results: list[ChunkSearchResult] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class UpsertChunksResult(BaseModel):
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


class EmbeddedChunk(BaseModel):
    chunk: PaperChunk
    vector: list[float]

    @field_validator("vector")
    @classmethod
    def vector_must_match_embedding_dimensions(cls, value: list[float]) -> list[float]:
        if len(value) != DEFAULT_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"vector must have {DEFAULT_EMBEDDING_DIMENSIONS} dimensions"
            )
        return value


class ChunkVectorSearchQuery(BaseModel):
    query_vector: list[float]
    session_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    limit: int = 5
    min_score: float | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_vector")
    @classmethod
    def query_vector_must_match_embedding_dimensions(
        cls,
        value: list[float],
    ) -> list[float]:
        if len(value) != DEFAULT_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"query_vector must have {DEFAULT_EMBEDDING_DIMENSIONS} dimensions"
            )
        return value

    @field_validator("limit")
    @classmethod
    def limit_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("limit must be positive")
        return value


class VectorSearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)
