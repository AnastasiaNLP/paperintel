"""
Live agent-to-agent Comparator test without batch graph integration.

Runs two normal single-paper graph invocations:
  Paper 0 -> completed single-paper state -> PaperSlot
  Paper 1 -> completed single-paper state -> PaperSlot

Then calls standalone comparator_agent({"papers": [slot0, slot1]}).

Uses real external services for the two single-paper runs:
  - Anthropic API
  - arXiv API
  - Semantic Scholar
  - GitHub API + HuggingFace API

No checkpointing and no production batch graph changes.

Run:
  python test_a2a_comparator_full.py
"""

import logging
import sys
import time
import uuid

import pytest

from agents.comparator import comparator_agent
from graph import create_app
from models.schemas import (
    BenchmarkResult,
    ComparisonReport,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)


pytestmark = pytest.mark.live

PAIR_LABEL = "DeepSeek-R1 vs Mistral 7B"
PAPER_0_URL = "https://arxiv.org/abs/2501.12948"
PAPER_1_URL = "https://arxiv.org/abs/2310.06825"
PIPELINE_TIMEOUT_SECONDS = 300
ALLOWED_ERROR_PREFIXES = (
    "Report warning:",
    "Benchmark initial parse failed",
    "Benchmark Sonnet fallback used",
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _initial_state(url: str) -> dict:
    return {
        "input_type": "url",
        "input_value": url,
        "papers": [],
        "metadata": None,
        "raw_text": None,
        "pdf_path": None,
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
        "ingestion_provenance": None,
        "comparison_report": None,
        "engineer_report": None,
        "full_markdown_report": None,
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "ingestion",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "messages": [],
        "errors": [],
        "cost_tracking": {},
    }


def _is_allowed_warning(error: str) -> bool:
    return any(error.startswith(prefix) for prefix in ALLOWED_ERROR_PREFIXES)


def _assert_single_paper_state(state: dict, url: str) -> None:
    errors = state.get("errors", [])
    non_warning_errors = [error for error in errors if not _is_allowed_warning(error)]

    assert state.get("processing_stage") == "completed", (
        f"{url}: expected completed, got {state.get('processing_stage')!r}; errors={errors}"
    )
    assert not non_warning_errors, f"{url}: non-warning errors: {non_warning_errors}"
    assert isinstance(state.get("metadata"), PaperMetadata)
    assert isinstance(state.get("method_extraction"), MethodExtraction)
    assert isinstance(state.get("benchmarks"), list)
    assert isinstance(state.get("production_readiness"), ProductionReadiness)
    assert isinstance(state.get("engineer_report"), EngineerReport)
    assert isinstance(state.get("full_markdown_report"), str)


def _run_single_paper(url: str, paper_index: int) -> dict:
    app = create_app(use_checkpointing=False)
    config = {
        "configurable": {"thread_id": f"a2a-{paper_index}-{uuid.uuid4()}"},
        "recursion_limit": 50,
    }

    print("=" * 72)
    print(f"Running single-paper pipeline for Paper {paper_index}: {url}")
    print("=" * 72)

    started = time.monotonic()
    state = app.invoke(_initial_state(url), config=config)
    elapsed = time.monotonic() - started
    print(f"Paper {paper_index} elapsed: {elapsed:.1f}s")

    if elapsed > PIPELINE_TIMEOUT_SECONDS:
        raise AssertionError(
            f"{url}: pipeline took {elapsed:.1f}s > {PIPELINE_TIMEOUT_SECONDS}s"
        )

    _assert_single_paper_state(state, url)
    _print_single_summary(state, paper_index)
    return state


def _build_paper_slot_from_state(state: dict, paper_index: int, input_url: str) -> PaperSlot:
    return PaperSlot(
        paper_index=paper_index,
        input_url=input_url,
        metadata=state.get("metadata"),
        method_extraction=state.get("method_extraction"),
        benchmarks=state.get("benchmarks", []) or [],
        production_readiness=state.get("production_readiness"),
        engineer_report=state.get("engineer_report"),
        markdown_report=state.get("full_markdown_report"),
        errors=state.get("errors", []) or [],
        completed=state.get("processing_stage") == "completed",
    )


def _assert_paper_slot(slot: PaperSlot) -> None:
    assert slot.completed is True
    assert slot.metadata is not None
    assert slot.method_extraction is not None
    assert slot.production_readiness is not None
    assert slot.engineer_report is not None
    assert slot.markdown_report
    assert isinstance(slot.benchmarks, list)


def _assert_comparator_result(result: dict) -> None:
    assert result.get("processing_stage") == "comparison_completed"
    report = result.get("comparison_report")
    markdown = result.get("comparison_markdown")

    assert isinstance(report, ComparisonReport), (
        f"comparison_report wrong type: {type(report).__name__}"
    )
    assert isinstance(markdown, str) and len(markdown) > 500
    assert len(report.papers_summary) == 2
    assert report.trade_offs.strip()
    assert report.overall_winner_reasoning.strip()
    assert report.winner_basis in {
        "readiness_dominant",
        "benchmark_dominant",
        "mixed",
        "no_clear_winner",
    }
    assert "# Paper Comparison" in markdown
    assert "## Benchmark Matrix" in markdown
    assert "## Recommendations" in markdown
    assert "## Overall" in markdown

    valid_indexes = {0, 1}
    if report.overall_winner_index is not None:
        assert report.overall_winner_index in valid_indexes

    for rec in report.recommendations:
        assert rec.recommended_paper_index in valid_indexes
        assert rec.constraint.strip()
        assert rec.reasoning.strip()

    rows_with_any_values = [
        row
        for row in report.comparison_matrix
        if any(value is not None for value in row.values_by_paper.values())
    ]
    assert rows_with_any_values, "comparison matrix has no benchmark values"
    if report.overall_winner_index is None:
        assert report.winner_basis == "no_clear_winner"
    elif report.rows_with_winner == 0:
        assert report.winner_basis == "readiness_dominant"

    # Serialization guard for later checkpoint/API integration.
    dumped = report.model_dump()
    assert dumped["papers_summary"]
    assert "comparison_matrix" in dumped


def _print_single_summary(state: dict, paper_index: int) -> None:
    metadata = state["metadata"]
    readiness = state["production_readiness"]
    report = state["engineer_report"]
    benchmarks = state.get("benchmarks", []) or []
    warnings = [error for error in state.get("errors", []) if _is_allowed_warning(error)]

    print(f"Paper {paper_index} title:          {metadata.title}")
    print(f"Paper {paper_index} arXiv ID:       {metadata.arxiv_id}")
    print(f"Paper {paper_index} benchmarks:     {len(benchmarks)}")
    print(f"Paper {paper_index} maturity:       {readiness.maturity_level}")
    print(f"Paper {paper_index} recommendation: {report.recommended_action}")
    if warnings:
        print(f"Paper {paper_index} warnings:       {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")
    print()


def _print_comparator_summary(result: dict) -> None:
    report = result["comparison_report"]
    markdown = result["comparison_markdown"]

    print("=" * 72)
    print("A2A COMPARATOR SUMMARY")
    print("=" * 72)
    print(f"Stage:                {result['processing_stage']}")
    print(f"Matrix rows:          {len(report.comparison_matrix)}")
    print(f"Rows with winners:    {len([row for row in report.comparison_matrix if row.winner_index is not None])}")
    print(f"Overall winner index: {report.overall_winner_index}")
    print(f"Winner basis:         {report.winner_basis}")
    print(f"Overlap ratio:        {report.benchmark_overlap_ratio:.2f}")
    print(f"Recommendations:      {len(report.recommendations)}")
    print(f"Markdown chars:       {len(markdown)}")
    print()
    print("Trade-offs:")
    print(report.trade_offs)
    print()
    print("Recommendations:")
    if not report.recommendations:
        print("_none_")
    else:
        for rec in report.recommendations:
            print(f"- {rec.constraint}: Paper {rec.recommended_paper_index}")
            print(f"  {rec.reasoning}")
    print()
    print("Overall:")
    print(report.overall_winner_reasoning)
    print("=" * 72)


def main() -> int:
    _configure_logging()
    print(f"=== A2A full Comparator test: {PAIR_LABEL} ===")
    print("No checkpointing. No batch graph integration.")
    print("Runs two full single-paper pipelines, then standalone comparator_agent.")
    print()

    paper_0_state = _run_single_paper(PAPER_0_URL, 0)
    paper_1_state = _run_single_paper(PAPER_1_URL, 1)

    paper_0_slot = _build_paper_slot_from_state(paper_0_state, 0, PAPER_0_URL)
    paper_1_slot = _build_paper_slot_from_state(paper_1_state, 1, PAPER_1_URL)
    _assert_paper_slot(paper_0_slot)
    _assert_paper_slot(paper_1_slot)

    print("=" * 72)
    print("Running standalone comparator_agent on finalized PaperSlots")
    print("=" * 72)
    comparator_result = comparator_agent(
        {
            "papers": [paper_0_slot, paper_1_slot],
            "processing_stage": "comparator",
            "errors": [],
        }
    )
    _assert_comparator_result(comparator_result)
    _print_comparator_summary(comparator_result)

    print("\nA2A full Comparator test: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
