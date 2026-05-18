"""Discovery graph for topic-driven paper search."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.research_strategist import research_strategist_agent
from agents.selection_advisor import selection_advisor_agent
from models.discovery_state import DiscoveryState
from services.searcher import Searcher


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _searcher(config: dict[str, Any] | None) -> Searcher | None:
    value = _configurable(config).get("searcher")
    if value is not None and hasattr(value, "search"):
        return value
    return None


def searcher_node(
    state: DiscoveryState,
    config: RunnableConfig,
) -> dict[str, Any]:
    searcher = _searcher(config)
    plan = state.get("discovery_plan")
    session_id = state.get("session_id")
    discovery_turn_id = state.get("discovery_turn_id")

    if searcher is None:
        return {
            "search_candidates": [],
            "search_warnings": ["Discovery searcher is not configured."],
        }
    if plan is None:
        return {
            "search_candidates": [],
            "search_warnings": ["Discovery plan is missing."],
        }
    if not session_id:
        return {
            "search_candidates": [],
            "search_warnings": ["session_id is missing."],
        }
    if not discovery_turn_id:
        return {
            "search_candidates": [],
            "search_warnings": ["discovery_turn_id is missing."],
        }

    result = searcher.search(
        session_id=session_id,
        discovery_turn_id=discovery_turn_id,
        plan=plan,
    )
    return {
        "search_candidates": result.candidates,
        "search_warnings": result.warnings,
    }


def finalize_discovery_node(state: DiscoveryState) -> dict[str, Any]:
    advice = state.get("selection_advice")
    return {
        "response_text": (
            advice.response_text
            if advice is not None
            else "I could not prepare a paper shortlist. Please try a more specific topic."
        ),
        "next_phase": "selection",
    }


def build_discovery_graph(checkpointer=None):
    graph = StateGraph(DiscoveryState)
    graph.add_node("research_strategist", research_strategist_agent)
    graph.add_node("searcher", searcher_node)
    graph.add_node("selection_advisor", selection_advisor_agent)
    graph.add_node("finalize_discovery", finalize_discovery_node)

    graph.set_entry_point("research_strategist")
    graph.add_edge("research_strategist", "searcher")
    graph.add_edge("searcher", "selection_advisor")
    graph.add_edge("selection_advisor", "finalize_discovery")
    graph.add_edge("finalize_discovery", END)

    return graph.compile(checkpointer=checkpointer)
