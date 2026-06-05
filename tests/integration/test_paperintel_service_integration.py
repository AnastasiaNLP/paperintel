from agents.agent_run_recorder import InMemoryAgentRunPersistence
from types import SimpleNamespace

from api.app_factory import _resolve_blob_store, create_paperintel_service
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore
from models.qa import AnswerDraft
from services.paperintel_service import PaperIntelService


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
