import pytest

from models.discovery import ResearchQuery
from services.arxiv_search_provider import ArxivSearchProvider


pytestmark = pytest.mark.live


def test_arxiv_search_provider_live_finds_transformer_paper():
    provider = ArxivSearchProvider(rate_limit_delay=0)

    results = provider.search(
        ResearchQuery(
            query="id:1706.03762",
            max_results=10,
        )
    )

    assert results
    assert results[0].arxiv_id == "1706.03762"
    assert "Attention Is All You Need" in results[0].title
