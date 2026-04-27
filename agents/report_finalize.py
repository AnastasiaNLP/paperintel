from copy import deepcopy

from agents.error_utils import fatal_error, is_batch
from models.schemas import PaperSlot
from models.state import PaperIntelState


_SCRATCH_RESET = {
    "metadata": None,
    "raw_text": None,
    "pdf_path": None,
    "text_by_page": None,
    "method_extraction": None,
    "benchmarks": [],
    "production_readiness": None,
    "engineer_report": None,
    "full_markdown_report": None,
    "ingestion_provenance": None,
    "confidence_scores": {},
    "needs_human_review": False,
    "human_review_reason": None,
    "paper_failed": False,
    "paper_failure_reason": None,
    "failed_node": None,
}


def _current_url(state: PaperIntelState | dict, paper_index: int) -> str:
    if is_batch(state):
        batch_urls = state.get("batch_urls")
        if not isinstance(batch_urls, list):
            raise ValueError("batch_urls must be a list in batch mode")
        if paper_index < 0 or paper_index >= len(batch_urls):
            raise IndexError(
                f"batch_urls index {paper_index} out of range for {len(batch_urls)} urls"
            )
        return str(batch_urls[paper_index])

    return str(state.get("input_value", ""))


def report_finalize_node(state: PaperIntelState | dict) -> dict:
    paper_index = state.get("current_paper_index")
    if not isinstance(paper_index, int) or paper_index < 0:
        return fatal_error(
            f"report_finalize received invalid current_paper_index: {paper_index!r}",
            "report_finalize",
        )

    engineer_report = state.get("engineer_report")
    if engineer_report is None:
        return fatal_error(
            "report_finalize requires engineer_report on success path",
            "report_finalize",
        )

    try:
        input_url = _current_url(state, paper_index)
    except Exception as exc:
        return fatal_error(str(exc), "report_finalize")

    benchmarks = state.get("benchmarks") or []
    if not isinstance(benchmarks, list):
        return fatal_error(
            f"report_finalize expected benchmarks list, got {type(benchmarks).__name__}",
            "report_finalize",
        )

    current_errors = list(state.get("errors", []) or [])

    slot = PaperSlot(
        paper_index=paper_index,
        input_url=input_url,
        metadata=state.get("metadata"),
        method_extraction=state.get("method_extraction"),
        benchmarks=deepcopy(benchmarks),
        production_readiness=state.get("production_readiness"),
        engineer_report=engineer_report,
        markdown_report=state.get("full_markdown_report"),
        errors=current_errors,
        completed=True,
    )

    return {
        "papers": [slot],
        "current_paper_index": paper_index + 1,
        "processing_stage": "report_finalize",
        "errors": [],
        **_SCRATCH_RESET,
    }
