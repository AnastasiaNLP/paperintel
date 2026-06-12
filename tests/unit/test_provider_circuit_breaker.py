import logging

import pytest

from conftest import assert_provider_failure_logged
from services.provider_circuit_breaker import PostgresProviderCircuitBreaker
from tools.circuit_breaker import CircuitBreakerOpenError


class FakeDecision:
    def __init__(self, *, allowed: bool, retry_after_seconds: float = 0.0) -> None:
        self.allowed = allowed
        self.retry_after_seconds = retry_after_seconds


class FakeRepository:
    def __init__(self, decision=None, error: Exception | None = None) -> None:
        self.decision = decision or FakeDecision(allowed=True)
        self.error = error
        self.calls = []

    def before_request(self, **kwargs):
        self.calls.append(("before_request", kwargs))
        if self.error is not None:
            raise self.error
        return self.decision

    def record_success(self, **kwargs):
        self.calls.append(("record_success", kwargs))
        if self.error is not None:
            raise self.error

    def record_failure(self, **kwargs):
        self.calls.append(("record_failure", kwargs))
        if self.error is not None:
            raise self.error


def test_postgres_provider_circuit_breaker_allows_request():
    repository = FakeRepository()
    breaker = PostgresProviderCircuitBreaker(repository)

    breaker.before_request(
        "arxiv",
        "api",
        failure_threshold=5,
        recovery_timeout_seconds=120,
    )

    assert repository.calls[0][0] == "before_request"


def test_postgres_provider_circuit_breaker_raises_open_error_on_deny():
    repository = FakeRepository(FakeDecision(allowed=False, retry_after_seconds=12.5))
    breaker = PostgresProviderCircuitBreaker(repository)

    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        breaker.before_request(
            "arxiv",
            "api",
            failure_threshold=5,
            recovery_timeout_seconds=120,
        )

    assert exc_info.value.service_name == "arxiv"
    assert exc_info.value.retry_after_seconds == 12.5


def test_postgres_provider_circuit_breaker_fail_open_on_repository_error(caplog):
    repository = FakeRepository(error=RuntimeError("db down"))
    breaker = PostgresProviderCircuitBreaker(repository)

    with caplog.at_level(logging.INFO, logger="services.provider_circuit_breaker"):
        breaker.before_request(
            "semantic_scholar",
            "api",
            failure_threshold=3,
            recovery_timeout_seconds=60,
        )

    assert repository.calls[0][0] == "before_request"
    message = caplog.records[-1].getMessage()
    assert_provider_failure_logged(
        message,
        provider="postgres",
        operation="provider_circuit_breaker.before_request",
        failure_class="provider_unavailable",
    )


def test_postgres_provider_circuit_breaker_can_fail_closed():
    repository = FakeRepository(error=RuntimeError("db down"))
    breaker = PostgresProviderCircuitBreaker(repository, fail_open=False)

    with pytest.raises(RuntimeError, match="db down"):
        breaker.record_failure(
            "semantic_scholar",
            "api",
            failure_threshold=3,
            recovery_timeout_seconds=60,
            failure_class="provider_unavailable",
        )
