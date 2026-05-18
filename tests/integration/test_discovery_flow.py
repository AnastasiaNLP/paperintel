from models.discovery import DiscoveryPlan, ResearchQuery, SearchCandidate, SelectionAdvice
from graph_discovery import build_discovery_graph, searcher_node


class FakeSearcher:
    def __init__(self, candidates=None, warnings=None):
        self.candidates = candidates or []
        self.warnings = warnings or []
        self.calls = []

    def search(self, *, session_id, discovery_turn_id, plan):
        self.calls.append(
            {
                "session_id": session_id,
                "discovery_turn_id": discovery_turn_id,
                "plan": plan,
            }
        )
        return type(
            "Result",
            (),
            {
                "candidates": self.candidates,
                "warnings": self.warnings,
            },
        )()


def _candidate(rank: int) -> SearchCandidate:
    arxiv_id = f"2401.0000{rank}"
    return SearchCandidate(
        id=f"candidate-{rank}",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=rank,
        title=f"Paper {rank}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
    )


def _plan() -> DiscoveryPlan:
    return DiscoveryPlan(
        topic="agent memory",
        queries=[ResearchQuery(query="agent memory", max_results=10)],
    )


def test_discovery_graph_happy_path(monkeypatch):
    candidates = [_candidate(1), _candidate(2)]

    def fake_strategist(state, config=None):
        return {
            "discovery_topic": "agent memory",
            "discovery_plan": _plan(),
            "agent_runs": [],
        }

    def fake_advisor(state, config=None):
        advice = SelectionAdvice(
            topic=state["discovery_topic"],
            response_text="Choose papers 1 or 2.",
            recommended_candidate_ids=["candidate-1"],
            candidate_count=len(state["search_candidates"]),
        )
        return {
            "selection_advice": advice,
            "response_text": advice.response_text,
            "agent_runs": [],
        }

    monkeypatch.setattr("graph_discovery.research_strategist_agent", fake_strategist)
    monkeypatch.setattr("graph_discovery.selection_advisor_agent", fake_advisor)
    searcher = FakeSearcher(candidates=candidates)

    graph = build_discovery_graph()
    result = graph.invoke(
        {
            "session_id": "session-1",
            "user_message": "Find papers about agent memory",
            "persona": "engineer",
            "discovery_turn_id": "turn-1",
            "agent_runs": [],
            "errors": [],
        },
        config={"configurable": {"searcher": searcher}},
    )

    assert result["discovery_topic"] == "agent memory"
    assert result["search_candidates"] == candidates
    assert result["selection_advice"].response_text == "Choose papers 1 or 2."
    assert result["response_text"] == "Choose papers 1 or 2."
    assert result["next_phase"] == "selection"
    assert searcher.calls[0]["discovery_turn_id"] == "turn-1"


def test_discovery_graph_empty_candidates_still_returns_advice(monkeypatch):
    def fake_strategist(state, config=None):
        return {
            "discovery_topic": "agent memory",
            "discovery_plan": _plan(),
            "agent_runs": [],
        }

    def fake_advisor(state, config=None):
        advice = SelectionAdvice(
            topic=state["discovery_topic"],
            response_text="I did not find candidate papers.",
            recommended_candidate_ids=[],
            candidate_count=0,
        )
        return {"selection_advice": advice, "response_text": advice.response_text}

    monkeypatch.setattr("graph_discovery.research_strategist_agent", fake_strategist)
    monkeypatch.setattr("graph_discovery.selection_advisor_agent", fake_advisor)

    result = build_discovery_graph().invoke(
        {
            "session_id": "session-1",
            "user_message": "Find papers about agent memory",
            "persona": "engineer",
            "discovery_turn_id": "turn-1",
        },
        config={"configurable": {"searcher": FakeSearcher(candidates=[])}},
    )

    assert result["search_candidates"] == []
    assert result["selection_advice"].candidate_count == 0
    assert result["next_phase"] == "selection"


def test_discovery_graph_search_warnings_propagate(monkeypatch):
    def fake_strategist(state, config=None):
        return {"discovery_topic": "agent memory", "discovery_plan": _plan()}

    def fake_advisor(state, config=None):
        advice = SelectionAdvice(
            topic="agent memory",
            response_text="Choose papers.",
            recommended_candidate_ids=[],
            candidate_count=0,
        )
        return {"selection_advice": advice}

    monkeypatch.setattr("graph_discovery.research_strategist_agent", fake_strategist)
    monkeypatch.setattr("graph_discovery.selection_advisor_agent", fake_advisor)

    result = build_discovery_graph().invoke(
        {
            "session_id": "session-1",
            "user_message": "Find papers",
            "persona": "engineer",
            "discovery_turn_id": "turn-1",
        },
        config={
            "configurable": {
                "searcher": FakeSearcher(warnings=["Search query failed"])
            }
        },
    )

    assert result["search_warnings"] == ["Search query failed"]


def test_searcher_node_handles_missing_searcher():
    result = searcher_node(
        {
            "session_id": "session-1",
            "discovery_turn_id": "turn-1",
            "discovery_plan": _plan(),
        },
        config={"configurable": {}},
    )

    assert result["search_candidates"] == []
    assert result["search_warnings"] == ["Discovery searcher is not configured."]
