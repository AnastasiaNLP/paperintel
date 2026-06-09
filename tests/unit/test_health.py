from types import SimpleNamespace

from services.health import HealthChecker


class FakeDbSession:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        provider_resilience_error: Exception | None = None,
    ) -> None:
        self.error = error
        self.provider_resilience_error = provider_resilience_error
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query):
        if self.error is not None:
            raise self.error
        if (
            self.provider_resilience_error is not None
            and "provider_" in str(query)
        ):
            raise self.provider_resilience_error
        self.executed.append(query)


class FakeQdrantClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def get_collections(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return []


class FakeBlobStore:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def ensure_bucket(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


class FakeQdrantStore:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        collection_config_error: Exception | None = None,
    ) -> None:
        self.client = FakeQdrantClient(error=error)
        self.collection_config_error = collection_config_error
        self.collection_config_calls = 0

    def check_connection(self) -> None:
        self.client.get_collections()

    def check_collection_config(self) -> None:
        self.collection_config_calls += 1
        if self.collection_config_error is not None:
            raise self.collection_config_error


def _settings(*, anthropic="anthropic-key", openai="openai-key"):
    return SimpleNamespace(
        anthropic_api_key=anthropic,
        openai_api_key=openai,
    )


def test_health_checker_reports_all_checks_ok():
    db = FakeDbSession()
    qdrant = FakeQdrantStore()
    checker = HealthChecker(
        session_factory=lambda: db,
        qdrant_store=qdrant,
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is True
    assert status.checks == {
        "postgres": "ok",
        "provider_resilience_store": "ok",
        "qdrant": "ok",
        "llm_provider": "configured",
        "openai_embeddings": "configured",
        "blob_store": "not_configured",
    }
    assert qdrant.client.calls == 1


def test_health_checker_reports_postgres_error():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(error=RuntimeError("db down")),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["postgres"] == "error:RuntimeError"
    assert status.checks["provider_resilience_store"] == "error:RuntimeError"


def test_health_checker_reports_provider_resilience_store_error():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(
            provider_resilience_error=RuntimeError("missing table")
        ),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["postgres"] == "ok"
    assert status.checks["provider_resilience_store"] == "error:RuntimeError"


def test_health_checker_reports_qdrant_error():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(error=RuntimeError("qdrant down")),
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["qdrant"] == "error:RuntimeError"


def test_health_checker_reports_qdrant_collection_config_error():
    qdrant = FakeQdrantStore(collection_config_error=ValueError("wrong dimensions"))
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=qdrant,
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["qdrant"] == "error:ValueError"
    assert qdrant.collection_config_calls == 1


def test_health_checker_reports_missing_llm_key():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(anthropic="", openai=""),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["llm_provider"] == "missing_api_key"
    assert status.checks["openai_embeddings"] == "missing_api_key"
    assert status.checks["blob_store"] == "not_configured"


def test_health_checker_reports_missing_optional_dependencies_as_not_configured():
    checker = HealthChecker()

    status = checker.check()

    assert status.healthy is False
    assert status.checks["postgres"] == "not_configured"
    assert status.checks["provider_resilience_store"] == "not_configured"
    assert status.checks["qdrant"] == "not_configured"
    assert status.checks["llm_provider"] == "not_configured"
    assert status.checks["openai_embeddings"] == "not_configured"
    assert status.checks["blob_store"] == "not_configured"


def test_health_checker_rejects_missing_required_blob_store():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(),
        blob_storage_required=True,
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["blob_store"] == "not_configured"



def test_health_checker_reports_configured_blob_store_ok():
    blob_store = FakeBlobStore()
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(),
        blob_store=blob_store,
    )

    status = checker.check()

    assert status.healthy is True
    assert status.checks["blob_store"] == "ok"
    assert blob_store.calls == 1


def test_health_checker_reports_blob_store_error():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(),
        settings=_settings(),
        blob_store=FakeBlobStore(error=RuntimeError("minio down")),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["blob_store"] == "error:RuntimeError"
