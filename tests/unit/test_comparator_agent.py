"""
Offline tests for agents.comparator.

These tests do NOT call:
  - Anthropic API
  - LangGraph
  - arXiv / Semantic Scholar / GitHub / HuggingFace

Run:
  python test_comparator_agent.py
"""

import json
from unittest.mock import patch

from agents.comparator import (
    _build_comparison_matrix,
    _build_matrix_stats,
    _build_papers_summary,
    _build_unique_rows_per_paper,
    _build_unique_tasks_per_paper,
    _eligible_indexes,
    _is_higher_better,
    _normalize_comparison_report,
    _normalize_condition_name,
    _normalize_metric_name,
    _normalize_task_name,
    _normalize_unit_name,
    _parse_claims,
    _select_representative_benchmark,
    comparator_agent,
)
from models.errors import error_message
from models.schemas import (
    BenchmarkResult,
    ComparisonReport,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)


def _benchmark(
    task: str,
    metric: str,
    value: float,
    *,
    unit: str | None = None,
    conditions: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task=task,
        metric=metric,
        value=value,
        unit=unit,
        conditions=conditions,
    )


def _paper(
    index: int,
    *,
    title: str | None = None,
    method: str | None = None,
    benchmarks: list[BenchmarkResult] | None = None,
    completed: bool = True,
    maturity: str = "experimental",
    action: str = "prototype",
) -> PaperSlot:
    title = title or f"Paper {index}"
    method = method or f"Method {index}"
    return PaperSlot(
        paper_index=index,
        input_url=f"https://arxiv.org/abs/test-{index}",
        metadata=PaperMetadata(
            title=title,
            authors=[f"Author {index}"],
            arxiv_id=f"test-{index}",
            published_date="2026-01-01",
            abstract="Test abstract.",
            categories=["cs.CL"],
            citation_count=10 + index,
        ),
        method_extraction=MethodExtraction(
            method_name=method,
            description="Test method.",
            novelty_claim="Test novelty.",
            key_components=["component"],
            compared_to=["baseline"],
            limitations_stated=["limitation"],
        ),
        benchmarks=benchmarks or [],
        production_readiness=ProductionReadiness(
            has_open_code=True,
            code_url=f"https://github.com/example/repo-{index}",
            huggingface_model=None,
            framework_integrations=["HuggingFace Transformers"],
            min_gpu_requirement=None,
            estimated_inference_cost=None,
            dependencies=["torch"],
            maturity_level=maturity,
            maturity_reasoning=f"Paper {index} readiness reasoning.",
        ),
        engineer_report=EngineerReport(
            executive_summary="Summary.",
            key_innovation="Innovation.",
            practical_implications="Implications.",
            implementation_difficulty="moderate",
            recommended_action=action,
            action_reasoning=f"Paper {index} action reasoning.",
        ),
        markdown_report=f"# {title}",
        completed=completed,
        errors=[],
    )


def _valid_claims(
    *,
    winner: int | None = 0,
    rec_index: int = 0,
    winner_basis: str = "mixed",
) -> str:
    return json.dumps(
        {
            "trade_offs": "Paper 0 has stronger aligned benchmarks; Paper 1 has lower deployment risk.",
            "winner_basis": winner_basis,
            "recommendations": [
                {
                    "constraint": "best accuracy",
                    "recommended_paper_index": rec_index,
                    "reasoning": "Paper 0 wins the comparable accuracy row.",
                }
            ],
            "overall_winner_index": winner,
            "overall_winner_reasoning": "Paper 0 has stronger benchmark evidence.",
        }
    )


def _normalize_report_for_test(
    claims: dict,
    papers: list[PaperSlot],
    matrix: list | None = None,
):
    matrix = matrix if matrix is not None else _build_comparison_matrix(papers)
    unique_tasks = _build_unique_tasks_per_paper(papers, matrix)
    unique_rows = _build_unique_rows_per_paper(papers, matrix)
    summary = _build_papers_summary(papers)
    stats = _build_matrix_stats(papers, matrix)
    return _normalize_comparison_report(
        claims,
        papers,
        matrix,
        unique_tasks,
        unique_rows,
        summary,
        stats,
    )


def test_normalize_task_aliases():
    assert _normalize_task_name("math500") == "math-500"
    assert _normalize_task_name("math 500") == "math-500"
    assert _normalize_task_name("MATH-500") == "math-500"
    assert _normalize_task_name("MMLU Pro") == "mmlu-pro"


def test_normalize_metric_aliases():
    assert _normalize_metric_name("acc") == "accuracy"
    assert _normalize_metric_name("pass @ 1") == "pass@1"
    assert _normalize_metric_name("EM") == "exact_match"


def test_normalize_condition_aliases():
    assert _normalize_condition_name("0-shot") == "zero-shot"
    assert _normalize_condition_name("zero shot") == "zero-shot"
    assert _normalize_condition_name("greedy decoding") == "greedy"


def test_normalize_unit_aliases():
    assert _normalize_unit_name("%") == "%"
    assert _normalize_unit_name("percent") == "%"
    assert _normalize_unit_name("percentage") == "%"
    assert _normalize_unit_name("milliseconds") == "ms"


def test_metric_direction_higher_is_better():
    assert _is_higher_better("accuracy") is True
    assert _is_higher_better("pass@1") is True
    assert _is_higher_better("win_rate") is True
    assert _is_higher_better("error_reduction") is True


def test_metric_direction_lower_is_better():
    assert _is_higher_better("perplexity") is False
    assert _is_higher_better("latency_ms") is False
    assert _is_higher_better("error_rate") is False
    assert _is_higher_better("wer") is False
    assert _is_higher_better("loss") is False


def test_representative_no_variants():
    representative, comparable, note = _select_representative_benchmark(
        [],
        higher_is_better=True,
    )
    assert representative is None
    assert comparable is True
    assert note == ""


def test_representative_one_variant():
    benchmark = _benchmark("MMLU", "accuracy", 70.0)
    representative, comparable, note = _select_representative_benchmark(
        [benchmark],
        higher_is_better=True,
    )
    assert representative == benchmark
    assert comparable is True
    assert note == ""


def test_representative_duplicates_same_condition_higher_better():
    low = _benchmark("MMLU", "accuracy", 70.0, conditions="zero-shot")
    high = _benchmark("MMLU", "accuracy", 75.0, conditions="0-shot")
    representative, comparable, note = _select_representative_benchmark(
        [low, high],
        higher_is_better=True,
    )
    assert representative == high
    assert comparable is False
    assert "best_value_by_metric_direction" in note


def test_representative_duplicates_same_condition_lower_better():
    slow = _benchmark("MMLU", "latency_ms", 120.0, conditions="greedy")
    fast = _benchmark("MMLU", "latency_ms", 80.0, conditions="greedy decoding")
    representative, comparable, note = _select_representative_benchmark(
        [slow, fast],
        higher_is_better=False,
    )
    assert representative == fast
    assert comparable is False
    assert "best_value_by_metric_direction" in note


def test_representative_duplicates_different_conditions_first_seen():
    first = _benchmark("MMLU", "accuracy", 70.0, conditions="zero-shot")
    second = _benchmark("MMLU", "accuracy", 75.0, conditions="5-shot")
    representative, comparable, note = _select_representative_benchmark(
        [first, second],
        higher_is_better=True,
    )
    assert representative == first
    assert comparable is False
    assert "first_seen" in note


def test_matrix_aligned_comparable_row_winner_and_margin():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0, unit="%")]),
        _paper(1, benchmarks=[_benchmark("mmlu", "acc", 70.4, unit="percent")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.task == "mmlu"
    assert row.metric == "accuracy"
    assert row.is_comparable is True
    assert row.values_by_paper[0] == 75.0
    assert row.values_by_paper[1] == 70.4
    assert row.winner_index == 0
    assert round(row.winner_margin or 0, 1) == 4.6


def test_matrix_lower_is_better_row_selects_smaller_value():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "latency_ms", 120.0, unit="ms")]),
        _paper(1, benchmarks=[_benchmark("MMLU", "latency_ms", 80.0, unit="milliseconds")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.higher_is_better is False
    assert row.is_comparable is True
    assert row.winner_index == 1
    assert row.winner_margin == 40.0


def test_matrix_condition_aliases_do_not_break_comparability():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0, conditions="zero shot")]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0, conditions="0-shot")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.is_comparable is True
    assert row.winner_index == 0


def test_matrix_true_condition_mismatch_is_non_comparable():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0, conditions="zero-shot")]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 80.0, conditions="5-shot")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.is_comparable is False
    assert row.winner_index is None
    assert row.winner_margin is None
    assert "different conditions" in (row.comparability_notes or "")


def test_matrix_unit_aliases_do_not_break_comparability():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0, unit="%")]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0, unit="percentage")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.is_comparable is True
    assert row.winner_index == 0


def test_matrix_unit_mismatch_is_non_comparable():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0, unit="%")]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0, unit="ms")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.is_comparable is False
    assert row.winner_index is None
    assert "different units" in (row.comparability_notes or "")


def test_matrix_missing_paper_value_is_explicit_none():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0)]),
        _paper(1, benchmarks=[]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.values_by_paper[0] == 75.0
    assert row.values_by_paper[1] is None
    assert row.duplicate_counts_by_paper[1] == 0
    assert row.winner_index is None


def test_matrix_duplicate_variants_counted_and_non_comparable():
    papers = [
        _paper(
            0,
            benchmarks=[
                _benchmark("MMLU", "accuracy", 75.0, conditions="zero-shot"),
                _benchmark("MMLU", "accuracy", 76.0, conditions="zero-shot"),
            ],
        ),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0, conditions="zero-shot")]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.duplicate_counts_by_paper[0] == 2
    assert row.duplicate_counts_by_paper[1] == 1
    assert row.values_by_paper[0] == 76.0
    assert row.is_comparable is False
    assert row.winner_index is None


def test_matrix_equal_values_have_no_winner_and_zero_margin():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0)]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 75.0)]),
    ]
    row = _build_comparison_matrix(papers)[0]
    assert row.is_comparable is True
    assert row.winner_index is None
    assert row.winner_margin == 0.0


def test_matrix_stats_for_overlap_and_wins():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0), _benchmark("AIME", "pass@1", 80.0)]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0), _benchmark("AIME", "pass@1", 75.0)]),
    ]
    matrix = _build_comparison_matrix(papers)
    stats = _build_matrix_stats(papers, matrix)
    assert stats["rows"] == 2
    assert stats["comparable_rows"] == 2
    assert stats["rows_with_winner"] == 2
    assert stats["benchmark_overlap_ratio"] == 1.0
    assert stats["wins_by_paper"] == {0: 2, 1: 0}


def test_unique_coverage_tasks_and_rows():
    papers = [
        _paper(
            0,
            benchmarks=[
                _benchmark("MMLU", "accuracy", 75.0),
                _benchmark("HumanEval", "pass@1", 65.0),
            ],
        ),
        _paper(
            1,
            benchmarks=[
                _benchmark("MMLU", "EM", 70.0),
                _benchmark("GSM8K", "accuracy", 80.0),
            ],
        ),
    ]
    matrix = _build_comparison_matrix(papers)
    unique_tasks = _build_unique_tasks_per_paper(papers, matrix)
    unique_rows = _build_unique_rows_per_paper(papers, matrix)
    assert "mmlu" not in unique_tasks[0]
    assert "mmlu" not in unique_tasks[1]
    assert "humaneval" in unique_tasks[0]
    assert "gsm8k" in unique_tasks[1]
    assert "mmlu/accuracy" in unique_rows[0]
    assert "mmlu/exact_match" in unique_rows[1]


def test_eligible_indexes_all_completed():
    assert _eligible_indexes([_paper(0, completed=True), _paper(1, completed=True)]) == {0, 1}


def test_eligible_indexes_mixed_completed_incomplete():
    assert _eligible_indexes([_paper(0, completed=True), _paper(1, completed=False)]) == {0}


def test_eligible_indexes_all_incomplete():
    assert _eligible_indexes([_paper(0, completed=False), _paper(1, completed=False)]) == {0, 1}


def test_parse_claims_clean_json():
    claims, error = _parse_claims(_valid_claims())
    assert error is None
    assert claims["overall_winner_index"] == 0


def test_parse_claims_fenced_json():
    claims, error = _parse_claims(f"```json\n{_valid_claims()}\n```")
    assert error is None
    assert claims["recommendations"][0]["constraint"] == "best accuracy"


def test_parse_claims_preamble_and_suffix():
    claims, error = _parse_claims(f"Here:\n{_valid_claims()}\nDone.")
    assert error is None
    assert claims["overall_winner_reasoning"]


def test_parse_claims_non_dict_rejected():
    claims, error = _parse_claims("[1, 2, 3]")
    assert claims is None
    assert "Expected JSON object" in error


def test_normalize_recommendations_invalid_indexes_dropped():
    papers = [_paper(0), _paper(1)]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "recommendations": [
                {"constraint": "best accuracy", "recommended_paper_index": 99, "reasoning": "bad"},
                {"constraint": "easiest to deploy", "recommended_paper_index": 0, "reasoning": "ok"},
            ],
            "overall_winner_index": 99,
            "overall_winner_reasoning": "bad winner",
        },
        papers,
    )
    assert len(report.recommendations) == 1
    assert report.recommendations[0].recommended_paper_index == 0
    assert report.overall_winner_index is None


def test_normalize_recommendations_duplicate_constraints_deduped():
    papers = [_paper(0), _paper(1)]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "recommendations": [
                {"constraint": "highest accuracy", "recommended_paper_index": 0, "reasoning": "a"},
                {"constraint": "top accuracy", "recommended_paper_index": 1, "reasoning": "b"},
            ],
            "overall_winner_reasoning": "ok",
        },
        papers,
    )
    assert len(report.recommendations) == 1
    assert report.recommendations[0].constraint == "best accuracy"


def test_normalize_recommendations_capped_at_five():
    papers = [_paper(0), _paper(1)]
    recs = [
        {"constraint": f"constraint {index}", "recommended_paper_index": 0, "reasoning": "ok"}
        for index in range(10)
    ]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "recommendations": recs,
            "overall_winner_reasoning": "ok",
        },
        papers,
    )
    assert len(report.recommendations) == 5


def test_normalize_report_missing_fields_use_fallbacks():
    papers = [_paper(0), _paper(1)]
    report = _normalize_report_for_test({}, papers)
    assert isinstance(report, ComparisonReport)
    assert report.trade_offs
    assert report.overall_winner_reasoning
    assert report.winner_basis == "no_clear_winner"


def test_normalize_report_incomplete_paper_cannot_be_winner_when_completed_exists():
    papers = [_paper(0, completed=True), _paper(1, completed=False)]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "recommendations": [
                {"constraint": "best accuracy", "recommended_paper_index": 1, "reasoning": "bad"},
                {"constraint": "easiest to deploy", "recommended_paper_index": 0, "reasoning": "ok"},
            ],
            "overall_winner_index": 1,
            "overall_winner_reasoning": "bad",
        },
        papers,
    )
    assert report.overall_winner_index is None
    assert report.winner_basis == "no_clear_winner"
    assert len(report.recommendations) == 1
    assert report.recommendations[0].recommended_paper_index == 0


def test_normalize_report_winner_basis_from_claim_when_valid():
    papers = [
        _paper(0, benchmarks=[_benchmark("MMLU", "accuracy", 75.0)]),
        _paper(1, benchmarks=[_benchmark("MMLU", "accuracy", 70.0)]),
    ]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "winner_basis": "benchmark_dominant",
            "overall_winner_index": 0,
            "overall_winner_reasoning": "wins benchmarks",
        },
        papers,
    )
    assert report.overall_winner_index == 0
    assert report.winner_basis == "benchmark_dominant"


def test_normalize_report_winner_basis_readiness_when_no_row_wins():
    papers = [_paper(0, maturity="experimental"), _paper(1, maturity="production_ready")]
    report = _normalize_report_for_test(
        {
            "trade_offs": "x",
            "overall_winner_index": 1,
            "overall_winner_reasoning": "better readiness",
        },
        papers,
    )
    assert report.overall_winner_index == 1
    assert report.rows_with_winner == 0
    assert report.winner_basis == "readiness_dominant"


@patch("agents.comparator._call_llm")
def test_entry_less_than_two_papers_skips(mock_call_llm):
    result = comparator_agent({"papers": [_paper(0)]})
    assert result["comparison_report"] is None
    assert result["comparison_markdown"] == ""
    assert "fewer than two papers" in error_message(result["errors"][0])
    mock_call_llm.assert_not_called()


@patch("agents.comparator._call_llm")
def test_entry_llm_failure_returns_deterministic_fallback(mock_call_llm):
    mock_call_llm.return_value = (None, "boom")
    result = comparator_agent({"papers": [_paper(0), _paper(1)]})
    assert result["processing_stage"] == "comparison_completed"
    assert isinstance(result["comparison_report"], ComparisonReport)
    assert result["comparison_report"].trade_offs
    assert "Paper Comparison" in result["comparison_markdown"]


@patch("agents.comparator._call_llm_repair")
@patch("agents.comparator._call_llm")
def test_entry_invalid_json_then_repair_success(mock_call_llm, mock_repair):
    mock_call_llm.return_value = ("not json", None)
    mock_repair.return_value = (
        _valid_claims(winner=1, rec_index=1, winner_basis="mixed"),
        None,
    )
    result = comparator_agent({"papers": [_paper(0), _paper(1)]})
    report = result["comparison_report"]
    assert report.overall_winner_index == 1
    assert report.recommendations[0].recommended_paper_index == 1
    mock_call_llm.assert_called_once()
    mock_repair.assert_called_once()


@patch("agents.comparator._call_llm_repair")
@patch("agents.comparator._call_llm")
def test_entry_invalid_json_and_repair_failure_returns_fallback(mock_call_llm, mock_repair):
    mock_call_llm.return_value = ("not json", None)
    mock_repair.return_value = ("still not json", None)
    result = comparator_agent({"papers": [_paper(0), _paper(1)]})
    report = result["comparison_report"]
    assert isinstance(report, ComparisonReport)
    assert report.overall_winner_index is None
    assert report.trade_offs
    mock_call_llm.assert_called_once()
    mock_repair.assert_called_once()


@patch("agents.comparator._call_llm")
def test_entry_markdown_escapes_pipes_and_newlines(mock_call_llm):
    mock_call_llm.return_value = (_valid_claims(), None)
    papers = [
        _paper(0, title="Paper | Zero", benchmarks=[_benchmark("Task | A", "accuracy", 75.0)]),
        _paper(1, title="Paper\nOne", benchmarks=[_benchmark("Task | A", "accuracy", 70.0)]),
    ]
    result = comparator_agent({"papers": papers})
    markdown = result["comparison_markdown"]
    assert "Paper \\| Zero" in markdown
    assert "Paper One" in markdown
    assert "task-\\|-a" in markdown


@patch("agents.comparator._call_llm")
def test_entry_comparison_report_model_dump_serializable(mock_call_llm):
    mock_call_llm.return_value = (_valid_claims(), None)
    result = comparator_agent({"papers": [_paper(0), _paper(1)]})
    dumped = result["comparison_report"].model_dump()
    json.dumps(dumped)
    assert "comparison_matrix" in dumped
    assert "winner_basis" in dumped


def main() -> None:
    tests = [
        (name, item)
        for name, item in sorted(globals().items())
        if name.startswith("test_") and callable(item)
    ]

    passed = 0
    for name, test in tests:
        test()
        passed += 1
        print(f"{name}: PASSED")

    print(f"\nComparator agent tests: PASSED ({passed} tests)")


if __name__ == "__main__":
    main()
