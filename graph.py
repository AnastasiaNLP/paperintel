import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.benchmark import benchmark_analyst_agent
from agents.extraction import extraction_agent
from agents.human_review import human_review_node
from agents.ingestion import ingestion_agent
from agents.supervisor import (
    route_after_benchmark,
    route_after_extraction,
    route_after_ingestion,
    supervisor_node,
)
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

SupervisorEntryRoute = Literal["ingestion", "error", "end"]


def _route_supervisor_entry(state: PaperIntelState) -> SupervisorEntryRoute:
    """
    Supervisor entry router.
    Routes initial state to the correct first agent.
    """
    stage = state.get("processing_stage", "")
    errors = state.get("errors", [])

    if stage == "failed":
        logger.warning("Supervisor -> error: %s", errors[-1] if errors else "unknown")
        return "error"

    if stage == "ingestion":
        return "ingestion"

    # topic_selection -> end (waiting for user choice)
    # everything else -> end (unexpected stage is not an error)
    logger.info("Supervisor -> end: stage=%s", stage)
    return "end"


def _error_node(state: PaperIntelState) -> dict:
    errors = state.get("errors", [])
    logger.error(
        "Pipeline error node. Last error: %s",
        errors[-1] if errors else "unknown",
    )
    return {"processing_stage": "failed"}


def build_graph() -> StateGraph:
    graph = StateGraph(PaperIntelState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("human_review", human_review_node)
    graph.add_node("error", _error_node)
    graph.add_node("benchmark", benchmark_analyst_agent)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor_entry,
        {
            "ingestion": "ingestion",
            "error": "error",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "ingestion",
        route_after_ingestion,
        {
            "extraction": "extraction",
            "end": END,
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "extraction",
        route_after_extraction,
        {
            "benchmark": "benchmark",
            "human_review": "human_review",
            "error": "error",
        },
    )

    graph.add_edge("human_review", "benchmark")

    graph.add_conditional_edges(
        "benchmark",
        route_after_benchmark,
        {
            "readiness": END,
            "error": "error",
        },
    )

    graph.add_edge("error", END)

    return graph


def create_app(use_checkpointing: bool = True):
    graph = build_graph()

    if use_checkpointing:
        try:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver
            from config.settings import settings

            conn = psycopg.connect(settings.postgres_url)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            app = graph.compile(
                checkpointer=checkpointer,
                interrupt_before=["human_review"],
            )
            logger.info("Graph compiled with PostgreSQL checkpointing")
            return app
        except Exception as exc:
            logger.warning(
                "PostgreSQL unavailable, running without checkpointing: %s", exc
            )

    app = graph.compile(interrupt_before=["human_review"])
    logger.info("Graph compiled without checkpointing")
    return app
