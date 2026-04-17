import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.ingestion import ingestion_agent
from agents.supervisor import (
    route_after_ingestion,
    supervisor_node,
)
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

Week1Route = Literal["ingestion", "error", "end"]


def _route_supervisor_week1(state: PaperIntelState) -> Week1Route:
    """
    supervisor router.
    Only ingestion | error | end are supported in the current graph.
    Full router enabled when extraction is wired in.
    """
    stage = state.get("processing_stage", "")
    errors = state.get("errors", [])

    if stage == "failed":
        logger.warning("Supervisor -> error: %s", errors[-1] if errors else "unknown")
        return "error"

    if stage == "ingestion":
        return "ingestion"

    # topic_selection -> end (expected: waiting for user choice)
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
    graph.add_node("error", _error_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor_week1,
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
            "extraction": END,  # replace with extraction node
            "end": END,
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
            checkpointer.setup()  # idempotent — safe to call on every start
            app = graph.compile(checkpointer=checkpointer)
            logger.info("Graph compiled with PostgreSQL checkpointing")
            return app

        except Exception as exc:
            logger.warning(
                "PostgreSQL unavailable, running without checkpointing: %s", exc
            )

    app = graph.compile()
    logger.info("Graph compiled without checkpointing")
    return app