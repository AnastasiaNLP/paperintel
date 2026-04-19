import logging

from models.state import PaperIntelState

logger = logging.getLogger(__name__)


def human_review_node(state: PaperIntelState) -> dict:
    """
    Human Review Node.

    Intended usage:
    - graph is compiled with interrupt_before=["human_review"]
    - execution pauses before this node
    - external code/user inspects checkpointed state
    - external code/user optionally updates method_extraction or other fields
    - app.invoke(None, config=config) resumes execution
    - this node marks review as completed and routes pipeline to benchmark

    This node does not perform review itself.
    It only records that execution resumed past the review gate.
    """
    reason = state.get("human_review_reason") or "Human review required"
    confidence = state.get("confidence_scores", {}).get("extraction")
    extraction = state.get("method_extraction")

    logger.warning(
        "Human review completed/resumed: reason=%s confidence=%s method=%s",
        reason,
        confidence,
        extraction.method_name if extraction else None,
    )

    return {
        "needs_human_review": False,
        "human_review_reason": None,
        "errors": state.get("errors", []),
        "processing_stage": "benchmark",
    }
