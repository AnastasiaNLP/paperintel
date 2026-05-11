from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    ChunkLocation,
    ChunkSearchQuery,
    ChunkSearchResult,
    ChunkSource,
    PaperChunk,
    UpsertChunksResult,
    VectorSearchHit,
)
from services.qdrant_store import chunk_payload
from services.retrieval_layer import PostgresQdrantRetrievalLayer


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.document_texts = []
        self.query_texts = []

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return _vector(0)

    def embed_documents(self, texts) -> list[list[float]]:
        self.document_texts.append(list(texts))
        return [_vector(index) for index, _ in enumerate(texts)]


class RecordingChunkRepository:
    def __init__(self, chunks_by_id=None) -> None:
        self.chunks_by_id = chunks_by_id or {}
        self.upserted = []
        self.requested_ids = []

    def upsert_many(self, chunks):
        self.upserted.extend(chunks)
        self.chunks_by_id.update({chunk.id: chunk for chunk in chunks})
        return UpsertChunksResult(inserted=len(chunks), updated=0, skipped=0)

    def get_many_by_ids(self, chunk_ids):
        self.requested_ids.append(list(chunk_ids))
        return [self.chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks_by_id]


class RecordingVectorStore:
    def __init__(self, hits=None) -> None:
        self.hits = hits or []
        self.ensure_calls = 0
        self.upserted = []
        self.search_queries = []

    def ensure_collection(self):
        self.ensure_calls += 1

    def upsert_chunks(self, embedded_chunks):
        self.upserted.extend(embedded_chunks)
        return UpsertChunksResult(inserted=len(embedded_chunks), updated=0, skipped=0)

    def search(self, query):
        self.search_queries.append(query)
        return self.hits


def _vector(index: int) -> list[float]:
    vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


def _chunk(chunk_id: str, paper_id: str = "2310.06825") -> PaperChunk:
    return PaperChunk(
        id=chunk_id,
        paper_id=paper_id,
        chunk_index=0,
        text=f"Chunk text for {chunk_id}",
        source=ChunkSource(
            paper_id=paper_id,
            session_id="session-1",
            arxiv_id=paper_id,
        ),
        location=ChunkLocation(page_start=1, page_end=1, section_title="Intro"),
    )


def test_postgres_qdrant_retrieval_layer_upserts_metadata_and_vectors():
    chunk = _chunk("2310.06825:chunk:0")
    repository = RecordingChunkRepository()
    vector_store = RecordingVectorStore()
    embeddings = RecordingEmbeddingProvider()
    layer = PostgresQdrantRetrievalLayer(
        chunk_repository=repository,
        vector_store=vector_store,
        embedding_provider=embeddings,
    )

    result = layer.upsert_chunks([chunk])

    assert result.inserted == 1
    assert repository.upserted == [chunk]
    assert embeddings.document_texts == [[chunk.text]]
    assert vector_store.ensure_calls == 1
    assert vector_store.upserted[0].chunk == chunk
    assert vector_store.upserted[0].vector == _vector(0)


def test_postgres_qdrant_retrieval_layer_searches_and_maps_persisted_chunks():
    chunk = _chunk("2310.06825:chunk:0")
    repository = RecordingChunkRepository({chunk.id: chunk})
    vector_store = RecordingVectorStore(
        [
            VectorSearchHit(
                chunk_id=chunk.id,
                paper_id=chunk.paper_id,
                score=0.91,
                payload=chunk_payload(chunk),
            )
        ]
    )
    embeddings = RecordingEmbeddingProvider()
    layer = PostgresQdrantRetrievalLayer(
        chunk_repository=repository,
        vector_store=vector_store,
        embedding_provider=embeddings,
    )

    results = layer.search_chunks(
        ChunkSearchQuery(
            query="retrieval evidence",
            session_id="session-1",
            paper_ids=["2310.06825"],
            limit=3,
            min_score=0.3,
        )
    )

    assert embeddings.query_texts == ["retrieval evidence"]
    assert repository.requested_ids == [[chunk.id]]
    assert vector_store.search_queries[0].session_id == "session-1"
    assert vector_store.search_queries[0].paper_ids == ["2310.06825"]
    assert vector_store.search_queries[0].limit == 3
    assert vector_store.search_queries[0].min_score == 0.3
    assert results[0].chunk == chunk
    assert results[0].rank == 1
    assert results[0].score == 0.91
    assert results[0].match_reason == "qdrant_vector_search"


def test_postgres_qdrant_retrieval_layer_falls_back_to_qdrant_payload():
    chunk = _chunk("2310.06825:chunk:0")
    repository = RecordingChunkRepository()
    vector_store = RecordingVectorStore(
        [
            VectorSearchHit(
                chunk_id=chunk.id,
                paper_id=chunk.paper_id,
                score=0.77,
                payload=chunk_payload(chunk),
            )
        ]
    )
    layer = PostgresQdrantRetrievalLayer(
        chunk_repository=repository,
        vector_store=vector_store,
        embedding_provider=RecordingEmbeddingProvider(),
    )

    results = layer.search_chunks(ChunkSearchQuery(query="fallback"))

    assert results[0].chunk.id == chunk.id
    assert results[0].chunk.text == chunk.text
    assert results[0].score == 0.77


def test_postgres_qdrant_retrieval_layer_assembles_evidence():
    chunk = _chunk("2310.06825:chunk:0")
    layer = PostgresQdrantRetrievalLayer(
        chunk_repository=RecordingChunkRepository(),
        vector_store=RecordingVectorStore(),
        embedding_provider=RecordingEmbeddingProvider(),
    )

    bundle = layer.assemble_evidence(
        "retrieval",
        [
            ChunkSearchResult(
                chunk=chunk,
                score=1.0,
                rank=1,
                match_reason="test",
            )
        ],
    )

    assert bundle.citations[0].chunk_id == chunk.id
    assert bundle.citations[0].page_start == 1
