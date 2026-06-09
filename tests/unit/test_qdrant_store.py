import sys
from types import SimpleNamespace
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
    QdrantCollectionMismatchError,
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


def test_embedded_chunk_accepts_non_default_vector_dimensions():
    vector = [0.0, 1.0]

    embedded = EmbeddedChunk(chunk=_chunk(), vector=vector)

    assert embedded.vector == vector
    with pytest.raises(ValidationError):
        EmbeddedChunk(chunk=_chunk(), vector=[])


def test_vector_search_query_accepts_non_default_dimensions_and_validates_limit():
    vector = [0.0, 1.0]

    query = ChunkVectorSearchQuery(query_vector=vector, limit=3)

    assert query.limit == 3
    with pytest.raises(ValidationError):
        ChunkVectorSearchQuery(query_vector=[], limit=3)
    with pytest.raises(ValidationError):
        ChunkVectorSearchQuery(query_vector=vector, limit=0)


def test_qdrant_store_rejects_wrong_vector_size_before_client_call():
    class ClientThatShouldNotBeCalled:
        def upsert(self, **kwargs):
            raise AssertionError("client should not be called")

    store = QdrantChunkStore(client=ClientThatShouldNotBeCalled(), vector_size=2)
    embedded = EmbeddedChunk(chunk=_chunk(), vector=[0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        store.upsert_chunks([embedded])


def test_qdrant_store_rejects_invalid_vector_size():
    with pytest.raises(ValueError, match="vector_size"):
        QdrantChunkStore(client=object(), vector_size=0)


def test_qdrant_store_from_url_preserves_configured_vector_size(monkeypatch):
    created = {}

    class FakeQdrantClient:
        def __init__(self, *, url, timeout):
            created["url"] = url
            created["timeout"] = timeout

    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        SimpleNamespace(QdrantClient=FakeQdrantClient),
    )

    store = QdrantChunkStore.from_url(
        url="http://qdrant.test",
        collection_name="paper_chunks_custom",
        vector_size=8,
        timeout=2.5,
    )

    assert created == {"url": "http://qdrant.test", "timeout": 2.5}
    assert store.collection_name == "paper_chunks_custom"
    assert store.vector_size == 8


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


def test_check_collection_config_allows_missing_collection_without_creating():
    class Collections:
        collections = []

    class MissingCollectionClient:
        def __init__(self):
            self.create_calls = 0

        def get_collections(self):
            return Collections()

        def create_collection(self, **kwargs):
            self.create_calls += 1

    client = MissingCollectionClient()
    store = QdrantChunkStore(client=client, vector_size=8)

    store.check_collection_config()

    assert client.create_calls == 0


def test_check_collection_config_rejects_non_default_vector_size_mismatch():
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
        def get_collections(self):
            return Collections()

        def get_collection(self, collection_name):
            assert collection_name == "paper_chunks"
            return CollectionInfo()

    store = QdrantChunkStore(client=ExistingCollectionClient(), vector_size=8)

    with pytest.raises(QdrantCollectionMismatchError, match="expected 8"):
        store.check_collection_config()


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
