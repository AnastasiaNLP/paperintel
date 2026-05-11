from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    ChunkLocation,
    ChunkSource,
    ChunkVectorSearchQuery,
    EmbeddedChunk,
    EvidenceArtifact,
    PaperChunk,
    UpsertChunksResult,
    VectorSearchHit,
)


DEFAULT_QDRANT_COLLECTION = "paper_chunks"
DEFAULT_QDRANT_DISTANCE = "Cosine"


class QdrantDependencyError(RuntimeError):
    pass


class QdrantCollectionMismatchError(RuntimeError):
    pass


class QdrantChunkStore:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
        distance: str = DEFAULT_QDRANT_DISTANCE,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

    @classmethod
    def from_url(
        cls,
        *,
        url: str,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        timeout: float = 10.0,
    ) -> "QdrantChunkStore":
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise QdrantDependencyError(
                "qdrant-client is required for QdrantChunkStore.from_url"
            ) from exc

        return cls(
            client=QdrantClient(url=url, timeout=timeout),
            collection_name=collection_name,
        )

    def ensure_collection(self) -> None:
        models = _qdrant_models()
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=getattr(models.Distance, self.distance.upper()),
                ),
            )
            return

        size, distance = self._collection_vector_config()
        if size != self.vector_size:
            raise QdrantCollectionMismatchError(
                f"Qdrant collection {self.collection_name!r} has vector size "
                f"{size}, expected {self.vector_size}"
            )
        if _normalize_distance(distance) != _normalize_distance(self.distance):
            raise QdrantCollectionMismatchError(
                f"Qdrant collection {self.collection_name!r} has distance "
                f"{distance!r}, expected {self.distance!r}"
            )

    def upsert_chunks(self, chunks: Sequence[EmbeddedChunk]) -> UpsertChunksResult:
        for embedded in chunks:
            _validate_vector(embedded.vector, self.vector_size)

        models = _qdrant_models()
        points = []
        for embedded in chunks:
            points.append(
                models.PointStruct(
                    id=qdrant_point_id(embedded.chunk.id),
                    vector=embedded.vector,
                    payload=chunk_payload(embedded.chunk),
                )
            )

        if not points:
            return UpsertChunksResult()

        self.client.upsert(collection_name=self.collection_name, points=points)
        return UpsertChunksResult(inserted=len(points), updated=0, skipped=0)

    def search(self, query: ChunkVectorSearchQuery) -> list[VectorSearchHit]:
        _validate_vector(query.query_vector, self.vector_size)
        query_filter = build_qdrant_filter(query)

        if hasattr(self.client, "search"):
            raw_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query.query_vector,
                query_filter=query_filter,
                limit=query.limit,
                score_threshold=query.min_score,
                with_payload=True,
            )
        else:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query.query_vector,
                query_filter=query_filter,
                limit=query.limit,
                score_threshold=query.min_score,
                with_payload=True,
            )
            raw_hits = result.points

        return [_hit_from_qdrant(hit) for hit in raw_hits]

    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(collection.name == self.collection_name for collection in collections)

    def _collection_vector_config(self) -> tuple[int, str]:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            vectors = next(iter(vectors.values()))
        return int(vectors.size), str(vectors.distance)


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def chunk_payload(chunk: PaperChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "paper_id": chunk.paper_id,
        "session_id": chunk.source.session_id,
        "paper_index": chunk.source.paper_index,
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "source": chunk.source.model_dump(mode="json"),
        "location": chunk.location.model_dump(mode="json"),
        "artifact_refs": [
            artifact.model_dump(mode="json") for artifact in chunk.artifact_refs
        ],
        "metadata": chunk.metadata,
        "embedding_model": chunk.embedding_model or DEFAULT_EMBEDDING_MODEL,
        "embedding_dimensions": chunk.embedding_dimensions,
    }


def chunk_from_payload(payload: dict[str, Any]) -> PaperChunk:
    return PaperChunk(
        id=str(payload["chunk_id"]),
        paper_id=str(payload["paper_id"]),
        chunk_index=int(payload["chunk_index"]),
        text=str(payload["text"]),
        chunk_type=payload.get("chunk_type", "text"),
        source=ChunkSource(**dict(payload.get("source") or {})),
        location=ChunkLocation(**dict(payload.get("location") or {})),
        artifact_refs=[
            EvidenceArtifact(**artifact)
            for artifact in list(payload.get("artifact_refs") or [])
        ],
        metadata=dict(payload.get("metadata") or {}),
        embedding_model=str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=int(
            payload.get("embedding_dimensions") or DEFAULT_EMBEDDING_DIMENSIONS
        ),
    )


def build_qdrant_filter(query: ChunkVectorSearchQuery) -> Any:
    must = []
    if query.session_id is not None:
        must.append(_field_condition("session_id", query.session_id))
    if query.paper_ids:
        must.append(_field_condition("paper_id", query.paper_ids))
    for key, value in query.filters.items():
        must.append(_field_condition(key, value))
    if not must:
        return None
    return _qdrant_models().Filter(must=must)


def _field_condition(key: str, value: Any) -> Any:
    models = _qdrant_models()
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=value))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _hit_from_qdrant(hit: Any) -> VectorSearchHit:
    payload = dict(hit.payload or {})
    return VectorSearchHit(
        chunk_id=str(payload.get("chunk_id") or hit.id),
        paper_id=str(payload.get("paper_id") or ""),
        score=float(hit.score),
        payload=payload,
    )


def _validate_vector(vector: Sequence[float], expected_dimensions: int) -> None:
    if len(vector) != expected_dimensions:
        raise ValueError(
            f"vector must have {expected_dimensions} dimensions, got {len(vector)}"
        )


def _normalize_distance(distance: Any) -> str:
    if hasattr(distance, "value"):
        distance = distance.value
    value = str(distance)
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.lower()


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise QdrantDependencyError(
            "qdrant-client is required for QdrantChunkStore operations"
        ) from exc
    return models
