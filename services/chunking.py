import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    ChunkLocation,
    ChunkSource,
    PaperChunk,
)
from models.schemas import PaperMetadata


class ChunkingConfig(BaseModel):
    target_chars: int = 2400
    overlap_chars: int = 300
    min_chunk_chars: int = 200
    include_abstract: bool = True

    @field_validator("target_chars", "min_chunk_chars")
    @classmethod
    def positive_sizes(cls, value: int) -> int:
        if value < 1:
            raise ValueError("chunk size values must be positive")
        return value

    @field_validator("overlap_chars")
    @classmethod
    def non_negative_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("overlap_chars must not be negative")
        return value


class ChunkingInput(BaseModel):
    metadata: PaperMetadata | None = None
    raw_text: str | None = None
    text_by_page: dict[int, str] | None = None
    session_id: str | None = None
    paper_index: int | None = None
    input_url: str | None = None


class ChunkingResult(BaseModel):
    paper_id: str
    chunks: list[PaperChunk] = Field(default_factory=list)
    skipped_reason: str | None = None


@dataclass(frozen=True)
class _TextSegment:
    text: str
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None
    section_title: str | None = None


class ChunkingService:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_paper(self, input: ChunkingInput) -> ChunkingResult:
        paper_id = resolve_paper_id(input)
        source = ChunkSource(
            paper_id=paper_id,
            paper_index=input.paper_index,
            session_id=input.session_id,
            input_url=input.input_url,
            title=input.metadata.title if input.metadata else None,
            arxiv_id=input.metadata.arxiv_id if input.metadata else None,
        )

        chunks: list[PaperChunk] = []
        if (
            self.config.include_abstract
            and input.metadata is not None
            and input.metadata.abstract.strip()
        ):
            chunks.append(
                self._make_chunk(
                    paper_id=paper_id,
                    chunk_index=len(chunks),
                    text=input.metadata.abstract,
                    chunk_type="abstract",
                    source=source,
                    location=ChunkLocation(section_title="Abstract"),
                )
            )

        for segment in self._segments(input):
            chunks.append(
                self._make_chunk(
                    paper_id=paper_id,
                    chunk_index=len(chunks),
                    text=segment.text,
                    chunk_type="text",
                    source=source,
                    location=ChunkLocation(
                        page_start=segment.page_start,
                        page_end=segment.page_end,
                        section_title=segment.section_title,
                        char_start=segment.char_start,
                        char_end=segment.char_end,
                    ),
                )
            )

        skipped_reason = None if chunks else "no_text_available"
        return ChunkingResult(paper_id=paper_id, chunks=chunks, skipped_reason=skipped_reason)

    def _segments(self, input: ChunkingInput) -> Iterable[_TextSegment]:
        if input.text_by_page:
            for page, page_text in sorted(input.text_by_page.items()):
                yield from self._split_text(page_text, page_start=page, page_end=page)
            return

        if input.raw_text:
            yield from self._split_text(input.raw_text, page_start=None, page_end=None)

    def _split_text(
        self,
        text: str,
        *,
        page_start: int | None,
        page_end: int | None,
    ) -> Iterable[_TextSegment]:
        normalized = _normalize_text(text)
        if len(normalized) < self.config.min_chunk_chars:
            if normalized:
                yield _TextSegment(
                    text=normalized,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=0,
                    char_end=len(normalized),
                    section_title=_detect_section_title(normalized),
                )
            return

        start = 0
        while start < len(normalized):
            end = min(start + self.config.target_chars, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind("\n\n", start, end)
                if boundary <= start + self.config.min_chunk_chars:
                    boundary = normalized.rfind(". ", start, end)
                if boundary > start + self.config.min_chunk_chars:
                    end = boundary + 1

            chunk_text = normalized[start:end].strip()
            if chunk_text:
                yield _TextSegment(
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=start,
                    char_end=end,
                    section_title=_detect_section_title(chunk_text),
                )

            if end >= len(normalized):
                break

            next_start = max(end - self.config.overlap_chars, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start

    def _make_chunk(
        self,
        *,
        paper_id: str,
        chunk_index: int,
        text: str,
        chunk_type: str,
        source: ChunkSource,
        location: ChunkLocation,
    ) -> PaperChunk:
        return PaperChunk(
            id=f"{paper_id}:chunk:{chunk_index}",
            paper_id=paper_id,
            chunk_index=chunk_index,
            text=text,
            chunk_type=chunk_type,
            source=source,
            location=location,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        )


def resolve_paper_id(input: ChunkingInput) -> str:
    if input.metadata and input.metadata.arxiv_id:
        return _strip_arxiv_version(input.metadata.arxiv_id)
    if input.input_url and input.input_url.strip():
        digest = hashlib.sha256(input.input_url.strip().encode("utf-8")).hexdigest()[:16]
        return f"url:{digest}"
    if input.paper_index is not None:
        return f"paper-index:{input.paper_index}"
    return "paper:unknown"


def _strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _detect_section_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip()
    if not first_line or len(first_line) > 90:
        return None
    if re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:/()&-]+$", first_line):
        return first_line
    return None
