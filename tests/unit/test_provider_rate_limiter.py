import logging

import pytest

from conftest import assert_provider_failure_logged
from services.provider_rate_limiter import PostgresProviderRateLimiter


class FakeReservation:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds


class FakeRepository:
    def __init__(self, reservation=None, error: Exception | None = None) -> None:
        self.reservation = reservation or FakeReservation(0)
        self.error = error
        self.calls = []

    def reserve_slot(self, *, provider, operation, interval_seconds):
        self.calls.append(
            {
                "provider": provider,
                "operation": operation,
                "interval_seconds": interval_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        return self.reservation


def test_postgres_provider_rate_limiter_reserves_and_sleeps():
    sleeps = []
    repository = FakeRepository(FakeReservation(1.25))
    limiter = PostgresProviderRateLimiter(repository, sleeper=sleeps.append)

    limiter.acquire("arxiv", "api", interval_seconds=3.2)

    assert repository.calls == [
        {"provider": "arxiv", "operation": "api", "interval_seconds": 3.2}
    ]
    assert sleeps == [1.25]


def test_postgres_provider_rate_limiter_fail_open_on_repository_error(caplog):
    sleeps = []
    repository = FakeRepository(error=RuntimeError("db down"))
    limiter = PostgresProviderRateLimiter(repository, sleeper=sleeps.append)

    with caplog.at_level(logging.INFO, logger="services.provider_rate_limiter"):
        limiter.acquire("semantic_scholar", "api", interval_seconds=1.2)

    assert repository.calls
    assert sleeps == [1.2]
    message = caplog.records[-1].getMessage()
    assert_provider_failure_logged(
        message,
        provider="postgres",
        operation="provider_rate_limiter.reserve_slot",
        failure_class="provider_unavailable",
    )


def test_postgres_provider_rate_limiter_can_fail_open_without_local_fallback():
    sleeps = []
    repository = FakeRepository(error=RuntimeError("db down"))
    limiter = PostgresProviderRateLimiter(
        repository,
        fallback_to_local=False,
        sleeper=sleeps.append,
    )

    limiter.acquire("semantic_scholar", "api", interval_seconds=1.2)

    assert repository.calls
    assert sleeps == []


def test_postgres_provider_rate_limiter_can_fail_closed():
    repository = FakeRepository(error=RuntimeError("db down"))
    limiter = PostgresProviderRateLimiter(
        repository,
        fail_open=False,
        sleeper=lambda seconds: None,
    )

    with pytest.raises(RuntimeError, match="db down"):
        limiter.acquire("semantic_scholar", "api", interval_seconds=1.2)
