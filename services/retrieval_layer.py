import re
from copy import deepcopy
from typing import Protocol, Sequence

from models.retrieval import (
    ChunkVectorSearchQuery,
    ChunkSearchQuery,
    ChunkSearchResult,
    CitationRef,
    EmbeddedChunk,
    EvidenceBundle,
    PaperChunk,
    UpsertChunksResult,
    VectorSearchHit,
)
from services.embeddings import EmbeddingProvider
from services.qdrant_store import QdrantChunkStore, chunk_from_payload
from storage.repositories import PostgresPaperChunkRepository


class RetrievalLayer(Protocol):
    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> UpsertChunksResult:
        ...

    def search_chunks(self, query: ChunkSearchQuery) -> list[ChunkSearchResult]:
        ...

    def assemble_evidence(
        self,
        query: str,
        results: Sequence[ChunkSearchResult],
        *,
        max_chunks: int = 5,
    ) -> EvidenceBundle:
        ...


class InMemoryRetrievalLayer:
    """
    Deterministic retrieval test double.

    This class intentionally does not generate embeddings. It provides the same
    contract shape as the future Qdrant-backed layer using lexical scoring.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, PaperChunk] = {}

    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> UpsertChunksResult:
        inserted = 0
        updated = 0
        skipped = 0

        for chunk in chunks:
            if not chunk.text.strip():
                skipped += 1
                continue

            if chunk.id in self._chunks:
                updated += 1
            else:
                inserted += 1
            self._chunks[chunk.id] = deepcopy(chunk)

        return UpsertChunksResult(inserted=inserted, updated=updated, skipped=skipped)

    def search_chunks(self, query: ChunkSearchQuery) -> list[ChunkSearchResult]:
        query_tokens = _tokens(query.query)
        results: list[ChunkSearchResult] = []

        for chunk in self._chunks.values():
            if query.session_id is not None and chunk.source.session_id != query.session_id:
                continue
            if query.paper_ids and chunk.paper_id not in query.paper_ids:
                continue

            score = _lexical_score(query_tokens, _tokens(chunk.text))
            if score <= 0:
                continue
            if query.min_score is not None and score < query.min_score:
                continue

            results.append(
                ChunkSearchResult(
                    chunk=deepcopy(chunk),
                    score=score,
                    rank=0,
                    match_reason="lexical_token_overlap",
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.paper_id,
                result.chunk.chunk_index,
                result.chunk.id,
            )
        )

        limited = results[: query.limit]
        for index, result in enumerate(limited, start=1):
            result.rank = index
        return limited

    def assemble_evidence(
        self,
        query: str,
        results: Sequence[ChunkSearchResult],
        *,
        max_chunks: int = 5,
    ) -> EvidenceBundle:
        selected = [deepcopy(result) for result in results[:max_chunks]]
        citations = [
            CitationRef(
                paper_id=result.chunk.paper_id,
                chunk_id=result.chunk.id,
                page_start=result.chunk.location.page_start,
                page_end=result.chunk.location.page_end,
                section_title=result.chunk.location.section_title,
                artifact_refs=deepcopy(result.chunk.artifact_refs),
            )
            for result in selected
        ]
        coverage_notes = []
        if not selected:
            coverage_notes.append("no_matching_chunks")

        return EvidenceBundle(
            query=query,
            results=selected,
            citations=citations,
            coverage_notes=coverage_notes,
        )


class PostgresQdrantRetrievalLayer:
    def __init__(
        self,
        *,
        chunk_repository: PostgresPaperChunkRepository,
        vector_store: QdrantChunkStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> UpsertChunksResult:
        if not chunks:
            return UpsertChunksResult()

        chunk_list = list(chunks)
        repository_result = self.chunk_repository.upsert_many(chunk_list)
        vectors = self.embedding_provider.embed_documents(
            [chunk.text for chunk in chunk_list]
        )
        embedded = [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunk_list, vectors)
        ]
        self.vector_store.ensure_collection()
        self.vector_store.upsert_chunks(embedded)
        return repository_result

    def search_chunks(self, query: ChunkSearchQuery) -> list[ChunkSearchResult]:
        query_vector = self.embedding_provider.embed_query(query.query)
        hits = self.vector_store.search(
            ChunkVectorSearchQuery(
                query_vector=query_vector,
                session_id=query.session_id,
                paper_ids=query.paper_ids,
                limit=query.limit,
                min_score=query.min_score,
                filters=query.filters,
            )
        )
        return _search_results_from_hits(hits, self.chunk_repository)

    def assemble_evidence(
        self,
        query: str,
        results: Sequence[ChunkSearchResult],
        *,
        max_chunks: int = 5,
    ) -> EvidenceBundle:
        return assemble_evidence_bundle(query, results, max_chunks=max_chunks)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def assemble_evidence_bundle(
    query: str,
    results: Sequence[ChunkSearchResult],
    *,
    max_chunks: int = 5,
) -> EvidenceBundle:
    selected = [deepcopy(result) for result in results[:max_chunks]]
    citations = [
        CitationRef(
            paper_id=result.chunk.paper_id,
            chunk_id=result.chunk.id,
            page_start=result.chunk.location.page_start,
            page_end=result.chunk.location.page_end,
            section_title=result.chunk.location.section_title,
            artifact_refs=deepcopy(result.chunk.artifact_refs),
        )
        for result in selected
    ]
    coverage_notes = []
    if not selected:
        coverage_notes.append("no_matching_chunks")

    return EvidenceBundle(
        query=query,
        results=selected,
        citations=citations,
        coverage_notes=coverage_notes,
    )


def _search_results_from_hits(
    hits: Sequence[VectorSearchHit],
    chunk_repository: PostgresPaperChunkRepository,
) -> list[ChunkSearchResult]:
    chunk_ids = [hit.chunk_id for hit in hits]
    persisted_chunks = chunk_repository.get_many_by_ids(chunk_ids)
    chunks_by_id = {chunk.id: chunk for chunk in persisted_chunks}

    results = []
    for index, hit in enumerate(hits, start=1):
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            chunk = chunk_from_payload(hit.payload)
        results.append(
            ChunkSearchResult(
                chunk=chunk,
                score=hit.score,
                rank=index,
                match_reason="qdrant_vector_search",
            )
        )
    return results
