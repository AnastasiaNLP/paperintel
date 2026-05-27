import pytest

from tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


def test_circuit_breaker_opens_after_threshold(monkeypatch):
    values = iter([10.0, 10.0, 11.0, 11.0])
    monkeypatch.setattr("tools.circuit_breaker.time.monotonic", lambda: next(values))
    breaker = CircuitBreaker(
        service_name="test_api",
        failure_threshold=2,
        recovery_timeout_seconds=30.0,
    )

    breaker.record_failure()
    assert breaker.state == "closed"

    breaker.record_failure()

    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        breaker.before_request()
    assert exc_info.value.service_name == "test_api"
    assert breaker.state == "open"


def test_circuit_breaker_transitions_to_half_open_after_timeout(monkeypatch):
    values = iter([10.0, 41.0, 41.0])
    monkeypatch.setattr("tools.circuit_breaker.time.monotonic", lambda: next(values))
    breaker = CircuitBreaker(
        service_name="test_api",
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
    )

    breaker.record_failure()

    breaker.before_request()
    assert breaker.state == "half_open"


def test_circuit_breaker_success_resets_failure_count():
    breaker = CircuitBreaker(
        service_name="test_api",
        failure_threshold=2,
        recovery_timeout_seconds=30.0,
    )

    breaker.record_failure()
    breaker.record_success()

    assert breaker.failure_count == 0
    assert breaker.state == "closed"
