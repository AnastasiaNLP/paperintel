import logging
from typing import Literal

from models.errors import ErrorCodes, make_error
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

SupervisorRoute = Literal[
    "ingestion",
    "extraction",
    "benchmark",
    "readiness",
    "report",
    "comparator",
    "paper_failure_finalize",
    "human_review",
    "end",
    "error",
]


def supervisor_node(state: PaperIntelState) -> dict:
    """
    Supervisor node for LangGraph routing.
    Reads the current state, logs high-level routing context,
    and fails fast if processing_stage is missing.
    """
    if "processing_stage" not in state:
        logger.error("Supervisor: processing_stage is missing from state")
        return {
            "errors": [
                make_error(
                    ErrorCodes.FATAL_ERROR,
                    "Supervisor: processing_stage is missing",
                    node="supervisor",
                    severity="fatal",
                    recoverable=False,
                )
            ],
            "processing_stage": "failed",
        }

    stage = state["processing_stage"]
    errors = state.get("errors", [])

    logger.info(
        "Supervisor: stage=%s errors=%d needs_review=%s",
        stage,
        len(errors),
        state.get("needs_human_review", False),
    )

    return {}


def route_after_supervisor(state: PaperIntelState) -> SupervisorRoute:
    """
    Conditional edge function for the supervisor.
    Routes based on processing_stage and review/error flags.
    """
    stage = state.get("processing_stage", "")
    needs_review = state.get("needs_human_review", False)
    errors = state.get("errors", [])

    if stage == "failed":
        logger.warning(
            "Supervisor routing to error: %s",
            errors[-1] if errors else "unknown",
        )
        return "error"

    if needs_review:
        logger.info("Supervisor routing to human_review")
        return "human_review"

    route_map: dict[str, SupervisorRoute] = {
        "ingestion": "ingestion",
        "extraction": "extraction",
        "benchmark": "benchmark",
        "readiness": "readiness",
        "report": "report",
        "topic_selection": "end",
    }

    route = route_map.get(stage, "error")
    logger.info("Supervisor routing: %s -> %s", stage, route)
    return route


def route_after_ingestion(state: PaperIntelState) -> SupervisorRoute:
    """Conditional edge after ingestion."""
    stage = state.get("processing_stage", "")

    if stage == "failed":
        return "error"
    if stage == "paper_failure_finalize":
        return "paper_failure_finalize"
    if stage == "topic_selection":
        return "end"
    if stage == "extraction":
        return "extraction"

    logger.warning("Unexpected stage after ingestion: %s", stage)
    return "error"


def route_after_extraction(state: PaperIntelState) -> SupervisorRoute:
    """
    Conditional edge after extraction.
    Low-confidence extraction routes to human review.
    """
    stage = state.get("processing_stage", "")
    needs_review = state.get("needs_human_review", False)

    if stage == "failed":
        return "error"
    if stage == "paper_failure_finalize":
        return "paper_failure_finalize"
    if needs_review:
        return "human_review"
    if stage == "benchmark":
        return "benchmark"

    logger.warning("Unexpected stage after extraction: %s", stage)
    return "error"


def route_after_readiness(state: PaperIntelState) -> SupervisorRoute:
    """
    Conditional edge after readiness.
    """
    stage = state.get("processing_stage", "")

    if stage == "failed":
        return "error"
    if stage == "paper_failure_finalize":
        return "paper_failure_finalize"
    if stage == "report":
        return "report"

    logger.warning("Unexpected stage after readiness: %s", stage)
    return "error"

def route_after_benchmark(state: PaperIntelState) -> SupervisorRoute:
    """Conditional edge after benchmark analyst."""
    stage = state.get("processing_stage", "")

    if stage == "failed":
        return "error"
    if stage == "paper_failure_finalize":
        return "paper_failure_finalize"
    if stage == "readiness":
        return "readiness"

    logger.warning("Unexpected stage after benchmark: %s", stage)
    return "error"


def route_after_finalize(state: PaperIntelState) -> SupervisorRoute:
    """
    Shared router after report_finalize and paper_failure_finalize.

    current_paper_index is expected to be incremented already by the finalize
    node that just ran.
    """
    current_index = state.get("current_paper_index", 0)
    total = state.get("total_papers", 1)
    papers = state.get("papers") or []

    if current_index < total:
        return "ingestion"
    if len(papers) >= 2:
        return "comparator"
    return "end"
