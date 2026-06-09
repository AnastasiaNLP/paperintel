import os
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config

from agents.agent_run_recorder import InMemoryAgentRunPersistence
import api.app_factory as app_factory
from api.app_factory import _resolve_blob_store, create_paperintel_service
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore
from models.artifacts import PaperWorkspace
from models.qa import AnswerDraft
from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    ChunkLocation,
    ChunkSearchQuery,
    ChunkSource,
    PaperChunk,
)
from services.arxiv_search_provider import ArxivSearchProvider
from services.paperintel_service import PaperIntelService
from services.provider_circuit_breaker import PostgresProviderCircuitBreaker
from services.provider_rate_limiter import PostgresProviderRateLimiter
from services.qdrant_store import QdrantChunkStore
from services.retrieval_layer import PostgresQdrantRetrievalLayer
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresAgentRunPersistence,
    PostgresPaperChunkRepository,
    PostgresPaperWorkspaceRepository,
    PostgresSessionStore,
    clear_foundation_tables,
)
import tools.arxiv_client as arxiv_client
import tools.semantic_scholar_client as semantic_scholar_client


class DeterministicEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return _vector(0)

    def embed_documents(self, texts) -> list[list[float]]:
        return [_vector(0) for _ in texts]


def _vector(index: int) -> list[float]:
    vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


def _database_url() -> str | None:
    return os.environ.get("PAPERINTEL_TEST_DATABASE_URL")


@pytest.fixture()
def postgres_session_factory():
    database_url = _database_url()
    if not database_url:
        pytest.skip("PAPERINTEL_TEST_DATABASE_URL is required for Postgres service tests")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    with factory() as db:
        clear_foundation_tables(db)

    yield factory

    with factory() as db:
        clear_foundation_tables(db)
    engine.dispose()


@pytest.fixture()
def qdrant_vector_store():
    collection_name = "paper_chunks_test_cache_reuse"
    try:
        store = QdrantChunkStore.from_url(
            url=os.environ.get("PAPERINTEL_TEST_QDRANT_URL", "http://localhost:6333"),
            collection_name=collection_name,
            timeout=10.0,
        )
        store.check_connection()
    except Exception as exc:
        pytest.skip(f"Qdrant is required for service cache reuse tests: {exc}")
    try:
        store.client.delete_collection(collection_name)
    except Exception:
        pass

    yield store

    try:
        store.client.delete_collection(collection_name)
    except Exception:
        pass


class FakeRunner:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        return self.result


class FakeRetrievalLayer:
    pass


class FakeChunkRepository:
    pass


class FakeVectorStore:
    def __init__(self, *, collection_name: str, vector_size: int) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout


class FakeBlobStore:
    def __init__(self) -> None:
        self.ensure_calls = 0

    def ensure_bucket(self):
        self.ensure_calls += 1


def _service(*, conversation_result=None, analysis_result=None):
    conversation_runner = FakeRunner(
        conversation_result
        or {
            "answer_draft": AnswerDraft(
                question="What is the method?",
                answer_text="It uses retrieval.",
                persona="engineer",
            ),
            "intent": "qa_factual",
            "referenced_paper_ids": ["1706.03762"],
        }
    )
    analysis_runner = FakeRunner(
        analysis_result
        or {
            "full_markdown_report": "# Analysis complete",
            "next_phase": "qa",
        }
    )
    handler = ChatHandler(
        store=InMemorySessionStore(),
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        agent_run_persistence=InMemoryAgentRunPersistence(),
    )
    return PaperIntelService(handler=handler), conversation_runner, analysis_runner


def test_service_full_session_lifecycle_with_chat_handler():
    service, conversation_runner, analysis_runner = _service()

    session = service.create_session(persona="engineer")
    analysis_result = service.analyze_paper(
        session.id,
        "https://arxiv.org/abs/1706.03762",
    )
    question_result = service.ask_question(session.id, "What is the method?")
    turns = service.list_turns(session.id)
    loaded = service.get_session(session.id)

    assert loaded.id == session.id
    assert analysis_result.intent == "analyze_paper"
    assert analysis_result.response_text == "# Analysis complete"
    assert question_result.intent == "qa_factual"
    assert question_result.response_text == "It uses retrieval."
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert len(analysis_runner.calls) == 1
    assert len(conversation_runner.calls) == 1


@pytest.mark.db
def test_service_reuses_arxiv_analysis_with_postgres_and_qdrant(
    postgres_session_factory,
    qdrant_vector_store,
):
    store = PostgresSessionStore(postgres_session_factory)
    artifact_repository = PostgresPaperWorkspaceRepository(postgres_session_factory)
    chunk_repository = PostgresPaperChunkRepository(postgres_session_factory)
    retrieval_layer = PostgresQdrantRetrievalLayer(
        chunk_repository=chunk_repository,
        vector_store=qdrant_vector_store,
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    analysis_runner = FakeRunner({"full_markdown_report": "# Should not run"})
    handler = ChatHandler(
        store=store,
        conversation_runner=FakeRunner({"response_text": "conversation"}),
        analysis_runner=analysis_runner,
        agent_run_persistence=PostgresAgentRunPersistence(postgres_session_factory),
        retrieval_layer=retrieval_layer,
    )
    service = PaperIntelService(
        handler=handler,
        artifact_repository=artifact_repository,
        paper_chunk_repository=chunk_repository,
    )
    source_session = store.create_session()
    target_session = service.create_session()
    source_workspace = artifact_repository.upsert_workspace(
        PaperWorkspace(
            session_id=source_session.id,
            paper_id="1706.03762",
            title="Attention Is All You Need",
            source_url="https://arxiv.org/abs/1706.03762",
            pipeline_stage="chunk_and_index",
            pipeline_version="v1",
            finalized_report_json={"recommended_action": "reuse"},
            method_extraction_json={"method_name": "Transformer"},
            readiness_json={"maturity_level": "production"},
            full_markdown_report="# Cached Transformer Report",
        )
    )
    chunk_repository.upsert_many(
        [
            PaperChunk(
                id="source-session-transformer-chunk-0",
                paper_id="1706.03762",
                chunk_index=0,
                text="Transformer attention retrieval evidence.",
                source=ChunkSource(
                    paper_id="1706.03762",
                    session_id=source_session.id,
                    arxiv_id="1706.03762",
                    title="Attention Is All You Need",
                ),
                location=ChunkLocation(
                    page_start=1,
                    page_end=1,
                    section_title="Abstract",
                ),
            )
        ]
    )

    result = service.analyze_paper(
        target_session.id,
        "https://arxiv.org/abs/1706.03762",
    )

    cloned_workspace = artifact_repository.get_workspace(
        target_session.id,
        "1706.03762",
    )
    target_chunks = chunk_repository.list_for_session_paper(
        target_session.id,
        "1706.03762",
    )
    search_results = retrieval_layer.search_chunks(
        ChunkSearchQuery(
            query="attention evidence",
            session_id=target_session.id,
            paper_ids=["1706.03762"],
            limit=5,
        )
    )
    loaded_session = store.require_session(target_session.id)
    turns = store.list_recent_turns(target_session.id, limit=10)

    assert cloned_workspace is not None
    assert cloned_workspace.id != source_workspace.id
    assert result.response_text == "# Cached Transformer Report"
    assert result.metadata == {
        "analysis_reused": True,
        "reuse_source": "paper_id",
    }
    assert result.artifact_refs == [f"paper_workspace:{cloned_workspace.id}"]
    assert loaded_session.phase == "qa"
    assert loaded_session.active_paper_ids == ["1706.03762"]
    assert len(target_chunks) == 1
    assert target_chunks[0].source.session_id == target_session.id
    assert search_results
    assert search_results[0].chunk.id == target_chunks[0].id
    assert search_results[0].chunk.source.session_id == target_session.id
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert analysis_runner.calls == []


def test_service_health_uses_basic_ok_without_checker():
    service, _, _ = _service()

    assert service.health().healthy is True
    assert service.health().checks == {"basic": "ok"}


def test_app_factory_creates_paperintel_service_with_injected_dependencies():
    conversation_runner = FakeRunner({"response_text": "conversation"})
    analysis_runner = FakeRunner({"response_text": "analysis"})
    retrieval_layer = FakeRetrievalLayer()
    paper_chunk_repository = FakeChunkRepository()
    blob_store = FakeBlobStore()

    service = create_paperintel_service(
        database_url="sqlite:///:memory:",
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        retrieval_layer=retrieval_layer,
        paper_chunk_repository=paper_chunk_repository,
        enable_health_checks=False,
        blob_store=blob_store,
    )

    assert service.handler.conversation_runner is conversation_runner
    assert service.handler.analysis_runner is analysis_runner
    assert service.handler.retrieval_layer is retrieval_layer
    assert service.paper_chunk_repository is paper_chunk_repository
    assert service.selected_candidate_resolver is not None
    assert service.candidate_repository is not None
    assert service.blob_store is blob_store
    assert service.blob_artifact_repository is not None
    assert blob_store.ensure_calls == 1
    assert service.health().checks == {"basic": "ok"}


def test_app_factory_reuses_chunk_repository_from_retrieval_layer():
    chunk_repository = FakeChunkRepository()
    retrieval_layer = FakeRetrievalLayer()
    retrieval_layer.chunk_repository = chunk_repository

    service = create_paperintel_service(
        database_url="sqlite:///:memory:",
        conversation_runner=FakeRunner({"response_text": "conversation"}),
        analysis_runner=FakeRunner({"response_text": "analysis"}),
        retrieval_layer=retrieval_layer,
        enable_health_checks=False,
        enable_blob_storage=False,
    )

    assert service.handler.retrieval_layer is retrieval_layer
    assert service.paper_chunk_repository is chunk_repository


def test_app_factory_wires_configured_embedding_model_and_dimensions(monkeypatch):
    from config.settings import settings as loaded_settings

    qdrant_calls = []

    def fake_from_url(*, url, collection_name, vector_size, timeout):
        qdrant_calls.append(
            {
                "url": url,
                "collection_name": collection_name,
                "vector_size": vector_size,
                "timeout": timeout,
            }
        )
        return FakeVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
        )

    monkeypatch.setattr(loaded_settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(loaded_settings, "qdrant_collection", "paper_chunks_custom")
    monkeypatch.setattr(loaded_settings, "qdrant_timeout", 7.5)
    monkeypatch.setattr(loaded_settings, "openai_api_key", "openai-key")
    monkeypatch.setattr(
        loaded_settings,
        "openai_embedding_model",
        "custom-embedding-model",
    )
    monkeypatch.setattr(loaded_settings, "openai_embedding_dimensions", 8)
    monkeypatch.setattr(loaded_settings, "openai_embedding_timeout", 12.0)
    monkeypatch.setattr(app_factory.QdrantChunkStore, "from_url", fake_from_url)
    monkeypatch.setattr(app_factory, "OpenAIEmbeddingProvider", FakeEmbeddingProvider)

    service = create_paperintel_service(
        database_url="postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel",
        conversation_runner=FakeRunner({"response_text": "conversation"}),
        analysis_runner=FakeRunner({"response_text": "analysis"}),
        discovery_runner=FakeRunner({"response_text": "discovery"}),
        paper_chunk_repository=FakeChunkRepository(),
        enable_health_checks=False,
        enable_blob_storage=False,
    )

    assert qdrant_calls == [
        {
            "url": "http://qdrant.test",
            "collection_name": "paper_chunks_custom",
            "vector_size": 8,
            "timeout": 7.5,
        }
    ]
    assert service.handler.retrieval_layer.vector_store.vector_size == 8
    assert (
        service.handler.retrieval_layer.embedding_provider.model
        == "custom-embedding-model"
    )
    assert service.handler.retrieval_layer.embedding_provider.dimensions == 8
    assert service.handler.retrieval_layer.embedding_provider.timeout == 12.0


def test_app_factory_health_includes_provider_resilience_store(postgres_session_factory):
    service = create_paperintel_service(
        database_url=_database_url(),
        conversation_runner=FakeRunner({"response_text": "conversation"}),
        analysis_runner=FakeRunner({"response_text": "analysis"}),
        retrieval_layer=FakeRetrievalLayer(),
        enable_health_checks=True,
        enable_blob_storage=False,
    )

    status = service.health()

    assert status.checks["postgres"] == "ok"
    assert status.checks["provider_resilience_store"] == "ok"

    search_provider = service.handler.searcher.provider
    assert isinstance(search_provider, ArxivSearchProvider)
    assert isinstance(search_provider.rate_limiter, PostgresProviderRateLimiter)
    assert isinstance(search_provider.circuit_breaker, PostgresProviderCircuitBreaker)
    assert arxiv_client._provider_rate_limiter is search_provider.rate_limiter
    assert semantic_scholar_client._provider_rate_limiter is search_provider.rate_limiter
    assert arxiv_client._provider_circuit_breaker is search_provider.circuit_breaker
    assert (
        semantic_scholar_client._provider_circuit_breaker
        is search_provider.circuit_breaker
    )


def test_app_factory_default_blob_storage_builds_from_settings(monkeypatch):
    created = FakeBlobStore()
    monkeypatch.setattr(
        "api.app_factory.S3BlobStore.from_config",
        lambda **kwargs: created,
    )

    service = create_paperintel_service(
        database_url="sqlite:///:memory:",
        conversation_runner=FakeRunner({"response_text": "conversation"}),
        analysis_runner=FakeRunner({"response_text": "analysis"}),
        discovery_runner=FakeRunner({"response_text": "discovery"}),
        retrieval_layer=FakeRetrievalLayer(),
        enable_health_checks=False,
    )

    assert service.blob_store is created
    assert service.blob_artifact_repository is not None
    assert created.ensure_calls == 1



def test_resolve_blob_store_supports_explicit_opt_out():
    assert _resolve_blob_store(
        blob_store=None,
        enable_blob_storage=False,
        settings=None,
    ) is None


def test_resolve_blob_store_builds_from_settings_and_bootstraps_bucket(monkeypatch):
    created = FakeBlobStore()
    calls = []

    def from_config(**kwargs):
        calls.append(kwargs)
        return created

    monkeypatch.setattr("api.app_factory.S3BlobStore.from_config", from_config)
    settings = SimpleNamespace(
        blob_storage_enabled=True,
        blob_s3_bucket="paperintel",
        blob_s3_endpoint_url="http://localhost:9000",
        blob_s3_region="us-east-1",
        blob_s3_access_key_id="paperintel",
        blob_s3_secret_access_key="secret",
    )

    resolved = _resolve_blob_store(
        blob_store=None,
        enable_blob_storage=None,
        settings=settings,
    )

    assert resolved is created
    assert created.ensure_calls == 1
    assert calls == [
        {
            "bucket_name": "paperintel",
            "endpoint_url": "http://localhost:9000",
            "region_name": "us-east-1",
            "access_key_id": "paperintel",
            "secret_access_key": "secret",
        }
    ]


def test_resolve_blob_store_explicit_enable_overrides_disabled_setting(monkeypatch):
    created = FakeBlobStore()
    monkeypatch.setattr(
        "api.app_factory.S3BlobStore.from_config",
        lambda **kwargs: created,
    )
    settings = SimpleNamespace(
        blob_storage_enabled=False,
        blob_s3_bucket="paperintel",
        blob_s3_endpoint_url="http://localhost:9000",
        blob_s3_region="us-east-1",
        blob_s3_access_key_id="paperintel",
        blob_s3_secret_access_key="secret",
    )

    resolved = _resolve_blob_store(
        blob_store=None,
        enable_blob_storage=True,
        settings=settings,
    )

    assert resolved is created
    assert created.ensure_calls == 1


def test_resolve_blob_store_propagates_bootstrap_failure():
    class FailingBlobStore:
        def ensure_bucket(self):
            raise RuntimeError("minio down")

    import pytest

    with pytest.raises(RuntimeError, match="minio down"):
        _resolve_blob_store(
            blob_store=FailingBlobStore(),
            enable_blob_storage=None,
            settings=None,
        )
