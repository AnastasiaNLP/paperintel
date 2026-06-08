from __future__ import annotations

import httpx
import pytest

from tools import semantic_scholar_client as s2
from tools.circuit_breaker import CircuitBreakerOpenError


@pytest.fixture(autouse=True)
def reset_s2_client():
    s2._last_request_at = 0.0
    s2.configure_provider_rate_limiter(None)
    s2.configure_provider_circuit_breaker(None)
    s2.reset_circuit_breaker()
    yield
    s2._last_request_at = 0.0
    s2.configure_provider_rate_limiter(None)
    s2.configure_provider_circuit_breaker(None)
    s2.reset_circuit_breaker()


class FakeClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response


class FakeRateLimiter:
    def __init__(self) -> None:
        self.calls = []

    def acquire(self, provider, operation, *, interval_seconds):
        self.calls.append(
            {
                "provider": provider,
                "operation": operation,
                "interval_seconds": interval_seconds,
            }
        )


class FakeCircuitBreaker:
    def __init__(self) -> None:
        self.calls = []

    def before_request(self, provider, operation, **kwargs):
        self.calls.append(("before_request", provider, operation, kwargs))

    def record_success(self, provider, operation, **kwargs):
        self.calls.append(("record_success", provider, operation, kwargs))

    def record_failure(self, provider, operation, **kwargs):
        self.calls.append(("record_failure", provider, operation, kwargs))


def test_get_paper_sends_api_key_header(monkeypatch):
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.com"),
        json={
            "paperId": "paper-1",
            "citationCount": 12,
            "influentialCitationCount": 3,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        },
    )
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-key")

    result = s2.get_paper("1706.03762")

    assert result["citation_count"] == 12
    assert client.calls[0]["headers"] == {"x-api-key": "secret-key"}


def test_get_paper_returns_empty_dict_on_rate_limit(monkeypatch):
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)
    monkeypatch.setattr(s2.get_paper.retry, "sleep", lambda seconds: None)

    assert s2.get_paper("1706.03762") == {}
    assert len(client.calls) == 3


def test_get_related_papers_returns_empty_list_on_forbidden(monkeypatch):
    response = httpx.Response(403, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)

    assert s2.get_related_papers("1706.03762") == []


def test_rate_limit_waits_between_process_local_requests(monkeypatch):
    s2._last_request_at = 0.0
    monotonic_values = iter([100.0, 100.0, 101.0, 101.2])
    sleeps = []

    monkeypatch.setattr(s2.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(s2.time, "sleep", sleeps.append)

    s2._rate_limit()
    s2._rate_limit()

    assert sleeps == [pytest.approx(0.2)]
    assert s2._last_request_at == 101.2


def test_rate_limit_uses_configured_provider_limiter(monkeypatch):
    limiter = FakeRateLimiter()
    s2.configure_provider_rate_limiter(limiter)
    monkeypatch.setattr(s2.time, "sleep", lambda seconds: None)

    s2._rate_limit()

    assert limiter.calls == [
        {
            "provider": "semantic_scholar",
            "operation": "api",
            "interval_seconds": s2.RATE_LIMIT_DELAY,
        }
    ]
    assert s2._last_request_at == 0.0


def test_get_paper_rate_limits_before_request(monkeypatch):
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.com"),
        json={"paperId": "paper-1", "citationCount": 1},
    )
    client = FakeClient(response)
    calls = []
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: calls.append("rate_limit"))

    result = s2.get_paper("1706.03762")

    assert result["citation_count"] == 1
    assert calls == ["rate_limit"]
    assert len(client.calls) == 1


def test_rate_limit_status_retries_before_degraded_fallback(monkeypatch):
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    calls = []
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: calls.append("rate_limit"))
    monkeypatch.setattr(s2.get_paper.retry, "sleep", lambda seconds: None)

    assert s2.get_paper("1706.03762") == {}
    assert calls == ["rate_limit", "rate_limit", "rate_limit"]


def test_s2_enrichment_failure_is_non_fatal(monkeypatch):
    from agents import ingestion

    def fail_get_paper(arxiv_id):
        raise RuntimeError("S2 down")

    monkeypatch.setattr(ingestion, "s2_get_paper", fail_get_paper)

    assert ingestion._enrich_s2("1706.03762") is None


def test_get_paper_opens_breaker_after_repeated_retryable_failures(monkeypatch):
    response = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            s2.get_paper.__wrapped__("1706.03762")

    with pytest.raises(CircuitBreakerOpenError):
        s2.get_paper.__wrapped__("1706.03762")

    assert len(client.calls) == 3


def test_get_paper_uses_configured_provider_circuit_breaker(monkeypatch):
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.com"),
        json={"paperId": "paper-1", "citationCount": 1},
    )
    breaker = FakeCircuitBreaker()
    monkeypatch.setattr(s2, "_client", FakeClient(response))
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)
    s2.configure_provider_circuit_breaker(breaker)

    assert s2.get_paper("1706.03762")["citation_count"] == 1
    assert breaker.calls[0][0:3] == ("before_request", "semantic_scholar", "api")
    assert breaker.calls[-1][0:3] == ("record_success", "semantic_scholar", "api")


def test_s2_external_failure_records_configured_provider_circuit_breaker(monkeypatch):
    response = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
    breaker = FakeCircuitBreaker()
    monkeypatch.setattr(s2, "_client", FakeClient(response))
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)
    s2.configure_provider_circuit_breaker(breaker)

    with pytest.raises(httpx.HTTPStatusError):
        s2.get_paper.__wrapped__("1706.03762")

    assert breaker.calls[-1][0:3] == ("record_failure", "semantic_scholar", "api")
    assert breaker.calls[-1][3]["failure_class"] == "provider_unavailable"


def test_s2_rate_limit_status_does_not_open_breaker(monkeypatch):
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            s2.get_paper.__wrapped__("1706.03762")

    assert s2._s2_breaker.failure_count == 0


def test_s2_rate_limit_status_is_retryable():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    assert s2._should_retry(exc) is True

    s2._record_s2_failure(exc)

    assert s2._s2_breaker.failure_count == 0


def test_s2_non_fatal_status_closes_half_open_breaker(monkeypatch):
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: None)
    monkeypatch.setattr(s2.get_paper.retry, "sleep", lambda seconds: None)
    s2._s2_breaker._state = "half_open"
    s2._s2_breaker._failure_count = 3

    assert s2.get_paper("1706.03762") == {}

    assert s2._s2_breaker.state == "closed"
    assert s2._s2_breaker.failure_count == 0
