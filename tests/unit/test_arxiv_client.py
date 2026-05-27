from datetime import datetime, timezone

import httpx
import pytest

from models.external_metadata import ArxivMetadataCacheEntry
from tools import arxiv_client


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
    arxiv_client._last_request_at = 0.0
    yield
    arxiv_client.configure_metadata_cache(None)
    arxiv_client._last_request_at = 0.0


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
