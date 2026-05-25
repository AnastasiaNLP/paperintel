from __future__ import annotations

import httpx

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
