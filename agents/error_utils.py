"""
Batch-aware error helpers.

Paper-level errors:
- In batch mode: record failure for the current paper and continue via
  paper_failure_finalize.
- In single-paper mode: fail the whole run.

Fatal errors:
- Always fail the whole run because graph/session invariants are broken.
"""

from typing import Optional


def is_batch(state) -> bool:
    """
    Batch mode is enabled only for an explicit list of 2+ URLs.

    This intentionally does not inspect input_type so single PDF and single URL
    continue to use the normal single-paper flow.
    """
    batch_urls = state.get("batch_urls")
    total_papers = state.get("total_papers", 1)

    return (
        isinstance(batch_urls, list)
        and len(batch_urls) > 1
        and isinstance(total_papers, int)
        and total_papers > 1
    )


def paper_error(state, message: str, node: str) -> dict:
    """
    Return a paper-scoped failure.

    In batch mode this routes to paper_failure_finalize so the batch can
    continue. In single-paper mode it becomes a terminal failure.
    """
    stage = "paper_failure_finalize" if is_batch(state) else "failed"
    return {
        "processing_stage": stage,
        "paper_failed": True,
        "paper_failure_reason": message,
        "failed_node": node,
        "errors": [message],
    }


def fatal_error(message: str, node: Optional[str] = None) -> dict:
    """
    Return a terminal session-level failure.

    Use this when graph/session invariants are broken and continuation would be
    unsafe, for example invalid batch indexes or corrupted state.
    """
    return {
        "processing_stage": "failed",
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": node,
        "errors": [message],
    }
