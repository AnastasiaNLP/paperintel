import pytest
from pydantic import ValidationError

from models.discovery import DiscoveryPlan, RawSearchResult, ResearchQuery, SearchCandidate, SelectionSet


def test_research_query_rejects_blank_query():
    with pytest.raises(ValidationError):
        ResearchQuery(query=" ")


def test_research_query_requires_positive_max_results():
    with pytest.raises(ValidationError):
        ResearchQuery(query="agent memory", max_results=0)


def test_raw_search_result_has_no_session_or_rank_fields():
    result = RawSearchResult(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        arxiv_id="1706.03762",
    )

    payload = result.model_dump()
    assert "session_id" not in payload
    assert "display_rank" not in payload


def test_search_candidate_display_rank_is_one_based():
    with pytest.raises(ValidationError):
        SearchCandidate(
            session_id="session-1",
            discovery_turn_id="turn-1",
            display_rank=0,
            title="Paper",
            url="https://arxiv.org/abs/1706.03762",
        )


def test_search_candidate_defaults_to_proposed_status():
    candidate = SearchCandidate(
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        title="Paper",
        url="https://arxiv.org/abs/1706.03762",
    )

    assert candidate.status == "proposed"


def test_discovery_plan_requires_topic():
    with pytest.raises(ValidationError):
        DiscoveryPlan(topic="")


def test_selection_set_ranks_are_one_based():
    with pytest.raises(ValidationError):
        SelectionSet(
            session_id="session-1",
            discovery_turn_id="turn-1",
            display_ranks=[1, 0],
        )
