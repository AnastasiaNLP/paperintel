import subprocess
import sys

from evaluation.benchmark_alignment_audit import (
    FAILURE_MATCHED,
    FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET,
    FAILURE_WRONG_METHOD_VARIANT,
    run_benchmark_alignment_audit,
)
from evaluation.golden_dataset import load_golden_records
from models.artifacts import PaperWorkspace


def _record(paper_id: str = "1706.03762"):
    record = next(
        record
        for record in load_golden_records("golden_dataset/paperintel_60_v2_2.jsonl")
        if record.paper_id == paper_id
    )
    return record.model_copy(update={"expected_benchmarks": [record.expected_benchmarks[0]]})


def _workspace(paper_id: str, benchmarks: list[dict]):
    return PaperWorkspace(
        session_id="session-1",
        paper_id=paper_id,
        source_url=f"https://arxiv.org/abs/{paper_id}",
        pipeline_stage="completed",
        benchmarks_json=benchmarks,
    )


def test_alignment_audit_reports_top_k_candidates_and_sources():
    record = _record()
    expected = record.expected_benchmarks[0]
    workspace = _workspace(
        record.paper_id,
        [
            {
                "task": "unrelated",
                "dataset": "other",
                "metric": expected.metric,
                "value": expected.value,
                "unit": expected.unit,
                "conditions_keywords": ["wrong"],
                "source_section": "Other",
                "source_table_or_figure": "Table 9",
            },
            {
                "task": expected.task,
                "dataset": expected.dataset,
                "metric": expected.metric,
                "value": expected.value,
                "unit": expected.unit,
                "conditions_keywords": expected.conditions_keywords,
                "source_section": expected.source_section,
                "source_table_or_figure": expected.source_table_or_figure,
                "evidence_anchor": expected.evidence_anchor.model_dump(),
            },
        ],
    )

    audit = run_benchmark_alignment_audit(
        records=[record],
        workspaces=[workspace],
        top_k=2,
    ).to_dict()

    paper = audit["papers"][0]
    row = paper["rows"][0]
    assert paper["matched_rows"] == 1
    assert paper["class_counts"] == {FAILURE_MATCHED: 1}
    assert paper["failure_counts"] == {}
    assert audit["summary"]["failure_counts"] == {}
    assert row["primary_failure_class"] == FAILURE_MATCHED
    assert len(row["actual_candidates"]) == 2
    assert row["actual_candidates"][0]["failure_class"] == FAILURE_MATCHED
    assert row["actual_candidates"][0]["actual_source"] == {
        "source_section": expected.source_section,
        "source_table_or_figure": expected.source_table_or_figure,
        "evidence_anchor": expected.evidence_anchor.model_dump(),
    }


def test_alignment_audit_classifies_value_metric_match_with_task_dataset_gap():
    record = _record()
    expected = record.expected_benchmarks[0]
    workspace = _workspace(
        record.paper_id,
        [
            {
                "task": "English-to-German translation",
                "dataset": "newstest2014",
                "metric": expected.metric,
                "value": expected.value,
                "unit": expected.unit,
                "conditions_keywords": expected.conditions_keywords,
                "source_table_or_figure": expected.source_table_or_figure,
            }
        ],
    )

    audit = run_benchmark_alignment_audit(
        records=[record],
        workspaces=[workspace],
    ).to_dict()

    candidate = audit["papers"][0]["rows"][0]["actual_candidates"][0]
    assert candidate["failure_class"] == FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET
    assert audit["summary"]["failure_counts"] == {
        FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET: 1
    }
    assert audit["summary"]["top_normalization_gaps"][0] == {
        "expected_task": expected.task,
        "expected_dataset": expected.dataset,
        "actual_task": "English-to-German translation",
        "actual_dataset": "newstest2014",
        "count": 1,
    }


def test_alignment_audit_classifies_wrong_method_variant():
    record = _record("2310.11511")
    expected = record.expected_benchmarks[0]
    workspace = _workspace(
        record.paper_id,
        [
            {
                "task": expected.task,
                "dataset": expected.dataset,
                "metric": expected.metric,
                "value": expected.value,
                "unit": expected.unit,
                "conditions_keywords": ["Llama2-7B", "Retrieve"],
                "source_table_or_figure": expected.source_table_or_figure,
            }
        ],
    )

    audit = run_benchmark_alignment_audit(
        records=[record],
        workspaces=[workspace],
    ).to_dict()

    candidate = audit["papers"][0]["rows"][0]["actual_candidates"][0]
    assert candidate["failure_class"] == FAILURE_WRONG_METHOD_VARIANT
    assert candidate["condition_keyword_coverage"]["missing"] == (
        expected.conditions_keywords
    )
    assert audit["papers"][0]["dominant_failure"] == FAILURE_WRONG_METHOD_VARIANT


def test_alignment_audit_summarizes_missing_workspace():
    record = _record()

    audit = run_benchmark_alignment_audit(
        records=[record],
        workspaces=[],
    ).to_dict()

    paper = audit["papers"][0]
    assert paper["workspace_found"] is False
    assert paper["actual_rows"] == 0
    assert paper["failure_counts"] == {"missing_due_to_extraction": 1}
    assert audit["summary"]["total_expected"] == 1


def test_alignment_audit_rejects_invalid_top_k():
    record = _record()

    try:
        run_benchmark_alignment_audit(
            records=[record],
            workspaces=[],
            top_k=0,
        )
    except ValueError as exc:
        assert str(exc) == "top_k must be >= 1"
    else:
        raise AssertionError("expected ValueError")


def test_alignment_audit_cli_rejects_zero_selected_records():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.benchmark_alignment_audit",
            "--golden",
            "golden_dataset/paperintel_60_v2_2.jsonl",
            "--workspaces",
            "does-not-need-to-exist.jsonl",
            "--paper-ids",
            "does-not-exist",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "ERROR filters selected zero records" in result.stdout


def test_alignment_audit_cli_rejects_invalid_top_k():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.benchmark_alignment_audit",
            "--golden",
            "golden_dataset/paperintel_60_v2_2.jsonl",
            "--workspaces",
            "does-not-need-to-exist.jsonl",
            "--top-k",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "ERROR --top-k must be >= 1" in result.stdout
