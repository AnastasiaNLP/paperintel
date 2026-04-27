from copy import deepcopy

from agents.error_utils import fatal_error
from agents.report_finalize import _SCRATCH_RESET, _current_url
from models.schemas import PaperSlot
from models.state import PaperIntelState


def _slot_errors(state: PaperIntelState | dict) -> list[str]:
    errors = list(state.get("errors", []) or [])
    failure_reason = state.get("paper_failure_reason")
    failed_node = state.get("failed_node")

    if failure_reason and failure_reason not in errors:
        errors.append(str(failure_reason))
    if failed_node:
        node_marker = f"Failed at node: {failed_node}"
        if node_marker not in errors:
            errors.append(node_marker)

    return errors


def paper_failure_finalize_node(state: PaperIntelState | dict) -> dict:
    paper_index = state.get("current_paper_index")
    if not isinstance(paper_index, int) or paper_index < 0:
        return fatal_error(
            f"paper_failure_finalize received invalid current_paper_index: {paper_index!r}",
            "paper_failure_finalize",
        )

    try:
        input_url = _current_url(state, paper_index)
    except Exception as exc:
        return fatal_error(str(exc), "paper_failure_finalize")

    benchmarks = state.get("benchmarks") or []
    if not isinstance(benchmarks, list):
        return fatal_error(
            f"paper_failure_finalize expected benchmarks list, got {type(benchmarks).__name__}",
            "paper_failure_finalize",
        )

    slot = PaperSlot(
        paper_index=paper_index,
        input_url=input_url,
        metadata=state.get("metadata"),
        method_extraction=state.get("method_extraction"),
        benchmarks=deepcopy(benchmarks),
        production_readiness=state.get("production_readiness"),
        engineer_report=state.get("engineer_report"),
        markdown_report=state.get("full_markdown_report"),
        errors=_slot_errors(state),
        completed=False,
    )

    return {
        "papers": [slot],
        "current_paper_index": paper_index + 1,
        "processing_stage": "paper_failure_finalize",
        "errors": [],
        **_SCRATCH_RESET,
    }
