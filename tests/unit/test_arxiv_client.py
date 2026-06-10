import logging
from datetime import datetime, timezone

import httpx
import pytest

from models.external_metadata import ArxivMetadataCacheEntry
from tools import arxiv_client
from tools.circuit_breaker import CircuitBreakerOpenError


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


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>Transformer paper.</summary>
    <author><name>Ashish Vaswani</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>
"""


class FakeMetadataCache:
    def __init__(self, entry=None, *, fail_get=False, fail_write=False):
        self.entry = entry
        self.fail_get = fail_get
        self.fail_write = fail_write
        self.get_calls = []
        self.successes = []
        self.errors = []

    def get(self, arxiv_id):
        self.get_calls.append(arxiv_id)
        if self.fail_get:
            raise RuntimeError("cache read failed")
        return self.entry

    def record_success(self, entry):
        if self.fail_write:
            raise RuntimeError("cache write failed")
        self.successes.append(entry)
        self.entry = entry
        return entry

    def record_error(self, arxiv_id, *, error_json):
        if self.fail_write:
            raise RuntimeError("cache write failed")
        entry = ArxivMetadataCacheEntry(
            arxiv_id=arxiv_id,
            last_error_json=error_json,
            error_count=1,
        )
        self.errors.append((arxiv_id, error_json))
        self.entry = entry
        return entry


@pytest.fixture(autouse=True)
def reset_arxiv_client():
    arxiv_client.configure_metadata_cache(None)
    arxiv_client.configure_provider_rate_limiter(None)
    arxiv_client.configure_provider_circuit_breaker(None)
    arxiv_client._last_request_at = 0.0
    arxiv_client.reset_circuit_breaker()
    yield
    arxiv_client.configure_metadata_cache(None)
    arxiv_client.configure_provider_rate_limiter(None)
    arxiv_client.configure_provider_circuit_breaker(None)
    arxiv_client._last_request_at = 0.0
    arxiv_client.reset_circuit_breaker()


def _response(text=FEED, *, status_code=200):
    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    return httpx.Response(status_code, text=text, request=request)


def test_get_metadata_returns_successful_cache_hit_without_network(monkeypatch):
    fetched_at = datetime(2017, 6, 12, tzinfo=timezone.utc)
    cache = FakeMetadataCache(
        ArxivMetadataCacheEntry(
            arxiv_id="1706.03762",
            title="Cached title",
            authors=["Cached Author"],
            abstract="Cached abstract.",
            published_date="2017-06-12",
            categories=["cs.CL"],
            fetched_at=fetched_at,
        )
    )
    arxiv_client.configure_metadata_cache(cache)

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be called on cache hit")

    monkeypatch.setattr(arxiv_client, "_get", fail_get)

    metadata = arxiv_client.get_metadata("1706.03762")

    assert metadata.title == "Cached title"
    assert metadata.authors == ["Cached Author"]
    assert cache.get_calls == ["1706.03762"]


def test_get_metadata_fetches_and_records_success_on_cache_miss(monkeypatch):
    cache = FakeMetadataCache()
    arxiv_client.configure_metadata_cache(cache)

    calls = []

    def fake_get(url, *, params=None):
        calls.append({"url": url, "params": params})
        return _response()

    monkeypatch.setattr(arxiv_client, "_get", fake_get)

    metadata = arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert metadata.title == "Attention Is All You Need"
    assert calls == [
        {
            "url": arxiv_client.ARXIV_API_URL,
            "params": {"id_list": "1706.03762"},
        }
    ]
    assert len(cache.successes) == 1
    assert cache.successes[0].arxiv_id == "1706.03762"
    assert cache.successes[0].source_url == "https://arxiv.org/abs/1706.03762"
    assert cache.successes[0].has_successful_fetch is True


def test_get_metadata_records_error_on_fetch_failure(monkeypatch):
    cache = FakeMetadataCache()
    arxiv_client.configure_metadata_cache(cache)

    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    response = httpx.Response(429, request=request)

    def fake_get(url, *, params=None):
        return response

    monkeypatch.setattr(arxiv_client, "_get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert len(cache.errors) == 1
    arxiv_id, error_json = cache.errors[0]
    assert arxiv_id == "1706.03762"
    assert error_json["code"] == "HTTPStatusError"
    assert error_json["http_status"] == 429


def test_get_metadata_emits_provider_failure_event_without_raw_body(
    monkeypatch,
    caplog,
):
    cache = FakeMetadataCache()
    arxiv_client.configure_metadata_cache(cache)

    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    response = httpx.Response(
        500,
        text="raw arxiv metadata body should not be logged",
        request=request,
    )

    def fake_get(url, *, params=None):
        return response

    monkeypatch.setattr(arxiv_client, "_get", fake_get)

    with caplog.at_level(logging.INFO, logger="services.provider_policy"):
        with pytest.raises(httpx.HTTPStatusError):
            arxiv_client.get_metadata.__wrapped__("1706.03762")

    message = next(
        record.getMessage()
        for record in caplog.records
        if "event=provider.failure" in record.getMessage()
    )
    assert 'provider="arxiv"' in message
    assert 'operation="metadata_or_search"' in message
    assert 'failure_class="provider_unavailable"' in message
    assert "retryable=true" in message
    assert "raw arxiv metadata body" not in message


def test_cache_failures_do_not_block_direct_fetch(monkeypatch):
    cache = FakeMetadataCache(fail_get=True, fail_write=True)
    arxiv_client.configure_metadata_cache(cache)
    monkeypatch.setattr(arxiv_client, "_get", lambda url, *, params=None: _response())

    metadata = arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert metadata.title == "Attention Is All You Need"


def test_rate_limit_waits_between_process_local_requests(monkeypatch):
    monotonic_values = iter([100.0, 100.0, 101.0, 103.2])
    sleeps = []

    monkeypatch.setattr(arxiv_client.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(arxiv_client.time, "sleep", sleeps.append)

    arxiv_client._rate_limit()
    arxiv_client._rate_limit()

    assert sleeps == [pytest.approx(2.2)]
    assert arxiv_client._last_request_at == pytest.approx(103.2)


def test_rate_limit_uses_configured_provider_limiter(monkeypatch):
    limiter = FakeRateLimiter()
    arxiv_client.configure_provider_rate_limiter(limiter)
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda seconds: None)

    arxiv_client._rate_limit()

    assert limiter.calls == [
        {
            "provider": "arxiv",
            "operation": "api",
            "interval_seconds": arxiv_client.RATE_LIMIT_DELAY,
        }
    ]
    assert arxiv_client._last_request_at == 0.0


def test_get_metadata_opens_breaker_after_repeated_external_failures(monkeypatch):
    cache = FakeMetadataCache()
    arxiv_client.configure_metadata_cache(cache)
    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    response = httpx.Response(500, request=request)
    calls = []

    def fake_get(url, *, params=None):
        calls.append(url)
        return response

    monkeypatch.setattr(arxiv_client, "_get", fake_get)

    for _ in range(5):
        with pytest.raises(httpx.HTTPStatusError):
            arxiv_client.get_metadata.__wrapped__("1706.03762")

    with pytest.raises(CircuitBreakerOpenError):
        arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert len(calls) == 5
    assert len(cache.errors) == 5


def test_get_metadata_uses_configured_provider_circuit_breaker(monkeypatch):
    breaker = FakeCircuitBreaker()
    arxiv_client.configure_provider_circuit_breaker(breaker)
    monkeypatch.setattr(arxiv_client, "_get", lambda url, *, params=None: _response())

    metadata = arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert metadata.title == "Attention Is All You Need"
    assert breaker.calls[0][0:3] == ("before_request", "arxiv", "api")
    assert breaker.calls[-1][0:3] == ("record_success", "arxiv", "api")


def test_external_failure_records_configured_provider_circuit_breaker(monkeypatch):
    breaker = FakeCircuitBreaker()
    arxiv_client.configure_provider_circuit_breaker(breaker)
    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    response = httpx.Response(500, request=request)
    monkeypatch.setattr(arxiv_client, "_get", lambda url, *, params=None: response)

    with pytest.raises(httpx.HTTPStatusError):
        arxiv_client.get_metadata.__wrapped__("1706.03762")

    assert breaker.calls[-1][0:3] == ("record_failure", "arxiv", "api")
    assert breaker.calls[-1][3]["failure_class"] == "provider_unavailable"


def test_rate_limit_status_is_retryable_but_not_breaker_failure():
    request = httpx.Request("GET", arxiv_client.ARXIV_API_URL)
    response = httpx.Response(429, request=request)
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    assert arxiv_client._should_retry_arxiv(exc) is True

    arxiv_client._record_arxiv_failure(exc)

    assert arxiv_client._arxiv_breaker.failure_count == 0


def test_paper_not_found_does_not_open_breaker(monkeypatch):
    empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""
    monkeypatch.setattr(
        arxiv_client,
        "_get",
        lambda url, *, params=None: _response(empty_feed),
    )

    with pytest.raises(arxiv_client.ArxivPaperNotFoundError):
        arxiv_client.get_metadata.__wrapped__("9999.99999")

    assert arxiv_client._arxiv_breaker.failure_count == 0


def test_local_value_errors_do_not_open_arxiv_breaker():
    arxiv_client._record_arxiv_failure(ValueError("PDF too large"))

    assert arxiv_client._arxiv_breaker.failure_count == 0


def test_paper_not_found_closes_half_open_breaker(monkeypatch):
    empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""
    monkeypatch.setattr(
        arxiv_client,
        "_get",
        lambda url, *, params=None: _response(empty_feed),
    )
    arxiv_client._arxiv_breaker._state = "half_open"
    arxiv_client._arxiv_breaker._failure_count = 5

    with pytest.raises(arxiv_client.ArxivPaperNotFoundError):
        arxiv_client.get_metadata.__wrapped__("9999.99999")

    assert arxiv_client._arxiv_breaker.state == "closed"
    assert arxiv_client._arxiv_breaker.failure_count == 0
