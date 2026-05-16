from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    ChunkSource,
    ChunkVectorSearchQuery,
    EmbeddedChunk,
    EvidenceArtifact,
    PaperChunk,
)
from services.qdrant_store import (
    QdrantChunkStore,
    chunk_payload,
    qdrant_point_id,
)


def _chunk() -> PaperChunk:
    return PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="Retrieval evidence for citations.",
        chunk_type="table",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id="session-1",
            paper_index=0,
            arxiv_id="2310.06825",
        ),
        artifact_refs=[
            EvidenceArtifact(
                paper_id="2310.06825",
                artifact_type="table",
                storage_ref="s3://paperintel/table-1.png",
            )
        ],
        metadata={"header_context": "Results"},
    )


def test_qdrant_point_id_is_deterministic_uuid5_from_chunk_id():
    chunk_id = "2310.06825:chunk:0"

    assert qdrant_point_id(chunk_id) == str(uuid5(NAMESPACE_URL, chunk_id))
    assert qdrant_point_id(chunk_id) == qdrant_point_id(chunk_id)


def test_chunk_payload_contains_retrieval_and_citation_context():
    payload = chunk_payload(_chunk())

    assert payload["chunk_id"] == "2310.06825:chunk:0"
    assert payload["paper_id"] == "2310.06825"
    assert payload["session_id"] == "session-1"
    assert payload["chunk_type"] == "table"
    assert payload["source"]["arxiv_id"] == "2310.06825"
    assert payload["artifact_refs"][0]["storage_ref"] == "s3://paperintel/table-1.png"
    assert payload["embedding_dimensions"] == DEFAULT_EMBEDDING_DIMENSIONS


def test_embedded_chunk_validates_vector_dimensions():
    vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS

    embedded = EmbeddedChunk(chunk=_chunk(), vector=vector)

    assert embedded.vector == vector
    with pytest.raises(ValidationError):
        EmbeddedChunk(chunk=_chunk(), vector=[0.0, 1.0])


def test_vector_search_query_validates_vector_dimensions_and_limit():
    vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS

    query = ChunkVectorSearchQuery(query_vector=vector, limit=3)

    assert query.limit == 3
    with pytest.raises(ValidationError):
        ChunkVectorSearchQuery(query_vector=[0.0], limit=3)
    with pytest.raises(ValidationError):
        ChunkVectorSearchQuery(query_vector=vector, limit=0)


def test_qdrant_store_rejects_wrong_vector_size_before_client_call():
    class ClientThatShouldNotBeCalled:
        def upsert(self, **kwargs):
            raise AssertionError("client should not be called")

    store = QdrantChunkStore(client=ClientThatShouldNotBeCalled(), vector_size=2)
    embedded = EmbeddedChunk(
        chunk=_chunk(),
        vector=[0.0] * DEFAULT_EMBEDDING_DIMENSIONS,
    )

    with pytest.raises(ValueError):
        store.upsert_chunks([embedded])


def test_ensure_collection_is_idempotent_for_existing_collection():
    class Collection:
        name = "paper_chunks"

    class Collections:
        collections = [Collection()]

    class VectorConfig:
        size = DEFAULT_EMBEDDING_DIMENSIONS
        distance = "Cosine"

    class Params:
        vectors = VectorConfig()

    class Config:
        params = Params()

    class CollectionInfo:
        config = Config()

    class ExistingCollectionClient:
        def __init__(self):
            self.create_calls = 0

        def get_collections(self):
            return Collections()

        def get_collection(self, collection_name):
            assert collection_name == "paper_chunks"
            return CollectionInfo()

        def create_collection(self, **kwargs):
            self.create_calls += 1

    client = ExistingCollectionClient()
    store = QdrantChunkStore(client=client)

    store.ensure_collection()
    store.ensure_collection()

    assert client.create_calls == 0


def test_qdrant_store_check_connection_uses_public_health_contract():
    class CollectionsClient:
        def __init__(self):
            self.calls = 0

        def get_collections(self):
            self.calls += 1
            return []

    client = CollectionsClient()
    store = QdrantChunkStore(client=client)

    store.check_connection()

    assert client.calls == 1
