from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArxivMetadataCacheEntry(BaseModel):
    arxiv_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published_date: str | None = None
    categories: list[str] = Field(default_factory=list)
    source_url: str | None = None
    fetched_at: datetime | None = None
    last_error_json: dict[str, Any] | None = None
    error_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def has_successful_fetch(self) -> bool:
        return self.fetched_at is not None
