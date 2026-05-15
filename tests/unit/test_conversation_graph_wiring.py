from langgraph.graph import END

from graph_conversation import (
    analysis_requested_response_node,
    build_conversation_graph,
    clarification_response_node,
    route_after_critic,
    route_after_intent,
)
from models.qa import RepairContext


def _compiled_graph_nodes(graph) -> set[str]:
    return set(graph.get_graph().nodes.keys())


def _compiled_graph_edges(graph) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in graph.get_graph().edges}


def test_graph_compiles_successfully():
    graph = build_conversation_graph()

    assert graph is not None


def test_graph_has_all_required_nodes():
    graph = build_conversation_graph()

    assert {
        "intent_router",
        "retrieval_planner",
        "answer_agent",
        "citation_critic",
        "clarification_response",
        "analysis_requested_response",
    }.issubset(_compiled_graph_nodes(graph))


def test_graph_entry_point_is_intent_router():
    graph = build_conversation_graph()

    assert ("__start__", "intent_router") in _compiled_graph_edges(graph)


def test_graph_has_terminal_edges_to_end():
    graph = build_conversation_graph()

    edges = _compiled_graph_edges(graph)
    assert ("clarification_response", END) in edges
    assert ("analysis_requested_response", END) in edges


def test_route_after_intent_qa_factual_goes_to_planner():
    assert route_after_intent({"intent": "qa_factual"}) == "retrieval_planner"


def test_route_after_intent_qa_math_goes_to_planner():
    assert route_after_intent({"intent": "qa_math"}) == "retrieval_planner"


def test_route_after_intent_qa_comparison_goes_to_planner():
    assert route_after_intent({"intent": "qa_comparison"}) == "retrieval_planner"


def test_route_after_intent_qa_followup_goes_to_planner():
    assert route_after_intent({"intent": "qa_followup"}) == "retrieval_planner"


def test_route_after_intent_clarification_flag_wins_over_qa_intent():
    assert (
        route_after_intent({"intent": "qa_factual", "needs_clarification": True})
        == "clarification_response"
    )


def test_route_after_intent_analyze_paper_goes_to_analysis_requested():
    assert route_after_intent({"intent": "analyze_paper"}) == (
        "analysis_requested_response"
    )


def test_route_after_intent_clarification_needed_goes_to_clarification():
    assert route_after_intent({"intent": "clarification_needed"}) == (
        "clarification_response"
    )


def test_route_after_intent_unclear_goes_to_clarification():
    assert route_after_intent({"intent": "unclear"}) == "clarification_response"


def test_route_after_intent_discover_goes_to_clarification():
    assert route_after_intent({"intent": "discover"}) == "clarification_response"


def test_route_after_intent_select_papers_goes_to_clarification():
    assert route_after_intent({"intent": "select_papers"}) == "clarification_response"


def test_route_after_intent_missing_intent_goes_to_clarification():
    assert route_after_intent({}) == "clarification_response"


def test_route_after_critic_no_repair_context_ends():
    assert route_after_critic({"critic_review": "accepted"}) == END


def test_route_after_critic_with_repair_context_loops_to_answer():
    repair_context = RepairContext(
        original_run_id="run-1",
        target_agent="answer_agent",
        instructions=["Remove unsupported claim."],
        iteration=1,
        critic_review_id="review-1",
    )

    assert route_after_critic({"repair_context": repair_context}) == "answer_agent"


def test_route_after_critic_repair_context_none_explicitly_ends():
    assert route_after_critic({"repair_context": None}) == END


def test_clarification_response_node_preserves_existing_question():
    assert clarification_response_node(
        {"clarification_question": "Which paper should I use?"}
    ) == {}


def test_clarification_response_node_sets_default_question():
    assert clarification_response_node({}) == {
        "needs_clarification": True,
        "clarification_question": "Please clarify what you want to do.",
    }


def test_analysis_requested_response_node_sets_needs_analysis():
    assert analysis_requested_response_node({}) == {
        "needs_analysis": True,
        "clarification_question": (
            "Please send the paper URL directly so I can analyze it."
        ),
    }
