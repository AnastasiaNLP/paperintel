"""Conversation graph for PaperIntel QA flow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.answer_agent import answer_agent
from agents.citation_critic import citation_critic_agent
from agents.intent_router import intent_router_agent
from agents.retrieval_planner import retrieval_planner_agent
from models.conversation_state import ConversationState


QA_INTENTS = frozenset({"qa_factual", "qa_math", "qa_comparison", "qa_followup"})


def clarification_response_node(state: ConversationState) -> dict[str, Any]:
    """Deterministic terminal node when conversation routing cannot proceed."""
    if state.get("clarification_question"):
        return {}
    return {
        "needs_clarification": True,
        "clarification_question": "Please clarify what you want to do.",
    }


def analysis_requested_response_node(state: ConversationState) -> dict[str, Any]:
    """Terminal node for messages routed as paper-analysis requests."""
    return {
        "needs_analysis": True,
        "clarification_question": (
            state.get("clarification_question")
            or "Please send the paper URL directly so I can analyze it."
        ),
    }


def discovery_requested_response_node(state: ConversationState) -> dict[str, Any]:
    """Terminal node for messages routed as paper-discovery requests."""
    topic = state.get("discovery_topic") or state.get("user_message")
    return {
        "needs_discovery": True,
        "discovery_topic": topic,
        "clarification_question": (
            state.get("clarification_question")
            or "Discovery is not wired yet. I can search for papers on this topic once discovery is configured."
        ),
    }


def route_after_intent(state: ConversationState) -> str:
    """Route by Intent Router output."""
    if state.get("needs_clarification"):
        return "clarification_response"

    intent = state.get("intent")
    if intent in QA_INTENTS:
        return "retrieval_planner"
    if intent == "analyze_paper":
        return "analysis_requested_response"
    if intent == "discover":
        return "discovery_requested_response"
    return "clarification_response"


def route_after_critic(state: ConversationState) -> str:
    """Route to repair when Citation Critic asks for another answer pass."""
    if state.get("repair_context") is not None:
        return "answer_agent"
    return END


def build_conversation_graph(checkpointer=None):
    """Build the LangGraph conversation QA graph."""
    graph = StateGraph(ConversationState)

    graph.add_node("intent_router", intent_router_agent)
    graph.add_node("retrieval_planner", retrieval_planner_agent)
    graph.add_node("answer_agent", answer_agent)
    graph.add_node("citation_critic", citation_critic_agent)
    graph.add_node("clarification_response", clarification_response_node)
    graph.add_node("analysis_requested_response", analysis_requested_response_node)
    graph.add_node("discovery_requested_response", discovery_requested_response_node)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "retrieval_planner": "retrieval_planner",
            "clarification_response": "clarification_response",
            "analysis_requested_response": "analysis_requested_response",
            "discovery_requested_response": "discovery_requested_response",
        },
    )

    graph.add_edge("retrieval_planner", "answer_agent")
    graph.add_edge("answer_agent", "citation_critic")

    graph.add_conditional_edges(
        "citation_critic",
        route_after_critic,
        {
            "answer_agent": "answer_agent",
            END: END,
        },
    )

    graph.add_edge("clarification_response", END)
    graph.add_edge("analysis_requested_response", END)
    graph.add_edge("discovery_requested_response", END)

    return graph.compile(checkpointer=checkpointer)
