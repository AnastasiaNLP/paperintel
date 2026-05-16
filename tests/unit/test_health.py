from types import SimpleNamespace

from services.health import HealthChecker


class FakeDbSession:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query):
        if self.error is not None:
            raise self.error
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


class FakeQdrantStore:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.client = FakeQdrantClient(error=error)

    def check_connection(self) -> None:
        self.client.get_collections()


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
        "qdrant": "ok",
        "llm_provider": "configured",
        "openai_embeddings": "configured",
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


def test_health_checker_reports_qdrant_error():
    checker = HealthChecker(
        session_factory=lambda: FakeDbSession(),
        qdrant_store=FakeQdrantStore(error=RuntimeError("qdrant down")),
        settings=_settings(),
    )

    status = checker.check()

    assert status.healthy is False
    assert status.checks["qdrant"] == "error:RuntimeError"


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


def test_health_checker_reports_missing_optional_dependencies_as_not_configured():
    checker = HealthChecker()

    status = checker.check()

    assert status.healthy is False
    assert status.checks["postgres"] == "not_configured"
    assert status.checks["qdrant"] == "not_configured"
    assert status.checks["llm_provider"] == "not_configured"
    assert status.checks["openai_embeddings"] == "not_configured"
