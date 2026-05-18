import httpx
import pytest

from models.discovery import ResearchQuery
from services.arxiv_search_provider import (
    ARXIV_API_URL,
    ArxivSearchProvider,
    build_search_query,
    canonical_abs_url,
    normalize_arxiv_id,
    normalize_query,
)


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <updated>2023-08-02T00:00:00Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>
      The dominant sequence transduction models are based on recurrent networks.
    </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1810.04805</id>
    <published>2018-10-11T00:00:00Z</published>
    <title>
      BERT: Pre-training of Deep Bidirectional Transformers
    </title>
    <summary>BERT abstract.</summary>
    <author><name>Jacob Devlin</name></author>
  </entry>
</feed>
"""


EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


MALFORMED_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
  </entry>
  <entry>
    <published>2018-01-01T00:00:00Z</published>
    <title>Missing ID</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.99999</id>
  </entry>
</feed>
"""


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params):
        self.calls.append({"url": url, "params": params})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True


def _response(text=FEED, *, status_code=200):
    request = httpx.Request("GET", ARXIV_API_URL)
    return httpx.Response(status_code, text=text, request=request)


def test_search_builds_arxiv_query_params():
    client = FakeClient([_response()])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    provider.search(ResearchQuery(query="attention transformer", max_results=5))

    assert client.calls == [
        {
            "url": ARXIV_API_URL,
            "params": {
                "search_query": "all:attention AND all:transformer",
                "max_results": 5,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        }
    ]


def test_search_normalizes_query_whitespace():
    assert normalize_query("  agent   memory\nretrieval  ") == "agent memory retrieval"

    client = FakeClient([_response()])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    provider.search(ResearchQuery(query="  agent   memory\nretrieval  ", max_results=5))

    assert client.calls[0]["params"]["search_query"] == (
        "all:agent AND all:memory AND all:retrieval"
    )


def test_search_builds_safe_multi_word_arxiv_query():
    assert build_search_query("retrieval augmented generation implementation") == (
        "all:retrieval AND all:augmented AND all:generation"
    )


def test_search_caps_free_text_query_to_three_terms():
    assert build_search_query(
        "retrieval augmented generation implementation benchmarks reproducibility"
    ) == "all:retrieval AND all:augmented AND all:generation"


def test_search_preserves_explicit_arxiv_field_query():
    assert build_search_query("id:1706.03762") == "id:1706.03762"
    assert build_search_query('ti:"Attention Is All You Need"') == (
        'ti:"Attention Is All You Need"'
    )

    client = FakeClient([_response()])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    provider.search(ResearchQuery(query="id:1706.03762", max_results=5))

    assert client.calls[0]["params"]["search_query"] == "id:1706.03762"


def test_search_caps_max_results():
    client = FakeClient([_response()])
    provider = ArxivSearchProvider(
        client=client,
        max_results_cap=25,
        rate_limit_delay=0,
    )

    provider.search(ResearchQuery(query="agent memory", max_results=100))

    assert client.calls[0]["params"]["max_results"] == 25


def test_search_returns_empty_on_empty_feed():
    provider = ArxivSearchProvider(
        client=FakeClient([_response(EMPTY_FEED)]),
        rate_limit_delay=0,
    )

    assert provider.search(ResearchQuery(query="no results")) == []


def test_search_parses_title_authors_abstract_dates():
    provider = ArxivSearchProvider(client=FakeClient([_response()]), rate_limit_delay=0)

    results = provider.search(ResearchQuery(query="transformer"))

    first = results[0]
    assert first.title == "Attention Is All You Need"
    assert first.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert first.abstract == (
        "The dominant sequence transduction models are based on recurrent networks."
    )
    assert first.published_at is not None
    assert first.year == 2017


def test_search_normalizes_arxiv_id_without_version():
    assert normalize_arxiv_id("http://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("https://arxiv.org/pdf/1706.03762v7.pdf") == "1706.03762"
    assert normalize_arxiv_id("1706.03762v7") == "1706.03762"


def test_search_builds_canonical_abs_url():
    assert canonical_abs_url("1706.03762") == "https://arxiv.org/abs/1706.03762"

    provider = ArxivSearchProvider(client=FakeClient([_response()]), rate_limit_delay=0)
    results = provider.search(ResearchQuery(query="transformer"))

    assert results[0].url == "https://arxiv.org/abs/1706.03762"


def test_search_skips_malformed_entries():
    provider = ArxivSearchProvider(
        client=FakeClient([_response(MALFORMED_FEED)]),
        rate_limit_delay=0,
    )

    results = provider.search(ResearchQuery(query="transformer"))

    assert len(results) == 1
    assert results[0].arxiv_id == "1706.03762"


def test_search_retries_transient_errors():
    timeout = httpx.TimeoutException("timeout")
    client = FakeClient([timeout, _response()])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    results = provider.search(ResearchQuery(query="transformer"))

    assert len(results) == 2
    assert len(client.calls) == 2


def test_search_retries_rate_limit_status():
    first = _response("rate limited", status_code=429)
    second = _response()
    client = FakeClient([first, second])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    results = provider.search(ResearchQuery(query="transformer"))

    assert len(results) == 2
    assert len(client.calls) == 2


def test_search_retries_server_error_status():
    first = _response("server error", status_code=503)
    second = _response()
    client = FakeClient([first, second])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    results = provider.search(ResearchQuery(query="transformer"))

    assert len(results) == 2
    assert len(client.calls) == 2


def test_search_raises_on_permanent_error_without_retry():
    response = _response("bad request", status_code=400)
    client = FakeClient([response])
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    with pytest.raises(httpx.HTTPStatusError):
        provider.search(ResearchQuery(query="bad"))

    assert len(client.calls) == 1


def test_provider_close_delegates_to_client():
    client = FakeClient([_response()])
    client.closed = False
    provider = ArxivSearchProvider(client=client, rate_limit_delay=0)

    provider.close()

    assert client.closed is True


def test_search_derives_year_from_published_at():
    provider = ArxivSearchProvider(client=FakeClient([_response()]), rate_limit_delay=0)

    results = provider.search(ResearchQuery(query="transformer"))

    assert results[0].year == results[0].published_at.year
