from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


CandidateStatus = Literal["proposed", "selected", "analyzed", "rejected"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchQuery(BaseModel):
    query: str
    max_results: int = 10
    source: str = "arxiv"

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("max_results")
    @classmethod
    def max_results_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_results must be positive")
        return value


class RawSearchResult(BaseModel):
    title: str
    url: str
    source: str = "arxiv"
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    published_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("title", "url")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class SearchCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    discovery_turn_id: str
    display_rank: int
    status: CandidateStatus = "proposed"
    title: str
    url: str
    source: str = "arxiv"
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("session_id", "discovery_turn_id", "title", "url")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("display_rank")
    @classmethod
    def display_rank_is_one_based(cls, value: int) -> int:
        if value < 1:
            raise ValueError("display_rank must be 1-based")
        return value


class DiscoveryPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    queries: list[ResearchQuery] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic must not be blank")
        return value


class SelectionSet(BaseModel):
    session_id: str
    discovery_turn_id: str
    selected_candidate_ids: list[str] = Field(default_factory=list)
    display_ranks: list[int] = Field(default_factory=list)

    @field_validator("display_ranks")
    @classmethod
    def ranks_are_one_based(cls, value: list[int]) -> list[int]:
        if any(rank < 1 for rank in value):
            raise ValueError("display ranks must be 1-based")
        return value


class SelectionAdvice(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    response_text: str
    recommended_candidate_ids: list[str] = Field(default_factory=list)
    candidate_count: int
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("topic", "response_text")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("candidate_count")
    @classmethod
    def candidate_count_must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("candidate_count must not be negative")
        return value
