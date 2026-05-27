from __future__ import annotations

import httpx
import pytest

from tools import semantic_scholar_client as s2


class FakeClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response


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

    assert s2.get_paper("1706.03762") == {}


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


def test_non_retryable_rate_limit_does_not_sleep_twice(monkeypatch):
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    client = FakeClient(response)
    calls = []
    monkeypatch.setattr(s2, "_client", client)
    monkeypatch.setattr(s2, "_rate_limit", lambda: calls.append("rate_limit"))

    assert s2.get_paper("1706.03762") == {}
    assert calls == ["rate_limit"]


def test_s2_enrichment_failure_is_non_fatal(monkeypatch):
    from agents import ingestion

    def fail_get_paper(arxiv_id):
        raise RuntimeError("S2 down")

    monkeypatch.setattr(ingestion, "s2_get_paper", fail_get_paper)

    assert ingestion._enrich_s2("1706.03762") is None
