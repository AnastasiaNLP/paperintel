from datetime import datetime, timezone

import pytest

from models.discovery import DiscoveryPlan, RawSearchResult, ResearchQuery, SearchCandidate
from services.searcher import Searcher


class FakeProvider:
    def __init__(self, results_by_query=None, failures=None):
        self.results_by_query = results_by_query or {}
        self.failures = failures or set()
        self.calls = []

    def search(self, query: ResearchQuery) -> list[RawSearchResult]:
        self.calls.append(query)
        if query.query in self.failures:
            raise RuntimeError("provider failed")
        return list(self.results_by_query.get(query.query, []))


class FakeRepository:
    def __init__(self):
        self.saved = []

    def upsert_many(self, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        self.saved = list(candidates)
        return list(candidates)


def _raw(
    title: str,
    *,
    url: str | None = None,
    arxiv_id: str | None = None,
    abstract: str | None = None,
    year: int | None = None,
    published_at=None,
) -> RawSearchResult:
    return RawSearchResult(
        title=title,
        url=url or f"https://arxiv.org/abs/{arxiv_id or title.casefold().replace(' ', '-')}",
        source="arxiv",
        authors=["Ada Lovelace"],
        year=year,
        arxiv_id=arxiv_id,
        abstract=abstract,
        published_at=published_at,
    )


def _plan(*queries: str) -> DiscoveryPlan:
    return DiscoveryPlan(
        topic="agent memory",
        queries=[ResearchQuery(query=query, max_results=10) for query in queries],
    )


def _searcher(provider: FakeProvider, repository: FakeRepository | None = None) -> Searcher:
    return Searcher(
        provider=provider,
        candidate_repository=repository or FakeRepository(),
    )


def test_searcher_calls_provider_for_each_query():
    provider = FakeProvider(
        {
            "agent memory": [_raw("Memory for Agents", arxiv_id="2401.1")],
            "long context agents": [_raw("Long Context Agents", arxiv_id="2401.2")],
        }
    )
    searcher = _searcher(provider)

    searcher.search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory", "long context agents"),
    )

    assert [query.query for query in provider.calls] == [
        "agent memory",
        "long context agents",
    ]


def test_searcher_deduplicates_by_arxiv_id():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw("Memory One", arxiv_id="2401.12345"),
                _raw("Memory Duplicate", arxiv_id="2401.12345"),
            ]
        }
    )
    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].arxiv_id == "2401.12345"


def test_searcher_deduplicates_by_canonical_url():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw(
                    "Memory One",
                    url="https://arxiv.org/pdf/2401.12345.pdf",
                    arxiv_id=None,
                ),
                _raw(
                    "Memory Duplicate",
                    url="https://arxiv.org/abs/2401.12345",
                    arxiv_id=None,
                ),
            ]
        }
    )
    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert len(result.candidates) == 1


def test_searcher_deduplicates_by_normalized_title_exact_match():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw("Memory for Agents!", arxiv_id=None, url="https://example.test/1"),
                _raw("memory for agents", arxiv_id=None, url="https://example.test/2"),
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert len(result.candidates) == 1


def test_searcher_does_not_fuzzy_deduplicate_similar_titles():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw("Memory for Agents", arxiv_id=None, url="https://example.test/1"),
                _raw("Memory in Agents", arxiv_id=None, url="https://example.test/2"),
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert len(result.candidates) == 2


def test_searcher_assigns_display_ranks_one_based():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw("Agent Memory A", arxiv_id="2401.1"),
                _raw("Agent Memory B", arxiv_id="2401.2"),
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert [candidate.display_rank for candidate in result.candidates] == [1, 2]


def test_searcher_sets_status_proposed():
    provider = FakeProvider({"agent memory": [_raw("Agent Memory", arxiv_id="2401.1")]})

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert result.candidates[0].status == "proposed"


def test_searcher_derives_year_from_published_at():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw(
                    "Agent Memory",
                    arxiv_id="2401.1",
                    published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert result.candidates[0].year == 2024


def test_searcher_scores_title_matches_above_abstract_matches():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw(
                    "Unrelated Systems",
                    arxiv_id="2401.1",
                    abstract="agent memory for long horizon planning",
                ),
                _raw(
                    "Agent Memory Architectures",
                    arxiv_id="2401.2",
                    abstract="systems paper",
                ),
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert result.candidates[0].title == "Agent Memory Architectures"


def test_searcher_preserves_reasons():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw("Agent Memory Architectures", arxiv_id="2401.1", year=2024),
            ]
        }
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert result.candidates[0].reasons
    assert any("title" in reason for reason in result.candidates[0].reasons)


def test_searcher_limits_to_top_candidates():
    provider = FakeProvider(
        {
            "agent memory": [
                _raw(f"Agent Memory {index}", arxiv_id=f"2401.{index}")
                for index in range(5)
            ]
        }
    )
    searcher = Searcher(
        provider=provider,
        candidate_repository=FakeRepository(),
        candidate_limit=3,
    )

    result = searcher.search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert len(result.candidates) == 3


def test_searcher_tolerates_one_query_failure():
    provider = FakeProvider(
        results_by_query={"working query": [_raw("Agent Memory", arxiv_id="2401.1")]},
        failures={"broken query"},
    )

    result = _searcher(provider).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("broken query", "working query"),
    )

    assert len(result.candidates) == 1
    assert len(result.warnings) == 1
    assert "broken query" in result.warnings[0]


def test_searcher_returns_empty_when_no_results():
    result = _searcher(FakeProvider()).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert result.candidates == []
    assert result.warnings == []


def test_searcher_persists_candidates():
    repository = FakeRepository()
    provider = FakeProvider(
        {"agent memory": [_raw("Agent Memory", arxiv_id="2401.1")]}
    )

    result = _searcher(provider, repository).search(
        session_id="session-1",
        discovery_turn_id="turn-1",
        plan=_plan("agent memory"),
    )

    assert repository.saved == result.candidates
    assert repository.saved[0].session_id == "session-1"
    assert repository.saved[0].discovery_turn_id == "turn-1"


def test_searcher_rejects_non_positive_candidate_limit():
    with pytest.raises(ValueError):
        Searcher(
            provider=FakeProvider(),
            candidate_repository=FakeRepository(),
            candidate_limit=0,
        )
