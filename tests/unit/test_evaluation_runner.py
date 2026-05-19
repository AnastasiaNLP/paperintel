import json

import pytest

from evaluation.golden_dataset import load_golden_records
from evaluation.runner import (
    EvaluationRunnerError,
    load_workspace_records,
    run_deterministic_evaluation,
    summarize_evaluation,
)
from models.artifacts import PaperWorkspace


def _workspace(record, *, complete: bool = True) -> PaperWorkspace:
    method = record.expected_method_extraction
    readiness = record.expected_readiness
    first_benchmark = record.expected_benchmarks[0]
    return PaperWorkspace(
        session_id="session-1",
        paper_id=record.paper_id,
        title=record.title,
        source_url=record.source_url,
        pipeline_stage="chunk_and_index",
        method_extraction_json={
            "method_name": method.method_name,
            "description": " ".join(method.description_keywords),
            "novelty_claim": " ".join(method.novelty_keywords),
            "key_components": method.key_components,
            "compared_to": method.compared_to,
            "limitations_stated": method.limitations_stated,
        },
        benchmarks_json=[
            {
                "task": first_benchmark.task,
                "metric": first_benchmark.metric,
                "value": first_benchmark.value,
                "unit": first_benchmark.unit,
                "conditions": " ".join(first_benchmark.conditions_keywords),
            }
        ]
        if not complete
        else [
            {
                "task": benchmark.task,
                "metric": benchmark.metric,
                "value": benchmark.value,
                "unit": benchmark.unit,
                "conditions": " ".join(benchmark.conditions_keywords),
            }
            for benchmark in record.expected_benchmarks
        ],
        readiness_json={
            "has_open_code": readiness.has_open_code,
            "code_url": readiness.code_url,
            "huggingface_model": readiness.huggingface_model,
            "framework_integrations": readiness.expected_framework_integrations,
            "min_gpu_requirement": readiness.min_gpu_requirement,
            "dependencies": readiness.dependencies,
            "maturity_level": readiness.maturity_level,
        },
        full_markdown_report=" ".join(record.expected_report_coverage.must_mention),
    )


def test_run_deterministic_evaluation_summarizes_scores_and_missing():
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:2]
    workspaces = [_workspace(records[0])]

    summary = run_deterministic_evaluation(records, workspaces)

    assert not summary.passed
    assert summary.total_records == 2
    assert summary.matched_workspaces == 1
    assert summary.missing_workspaces == ["2005.11401"]
    assert summary.average_score == 1.0
    assert summary.check_averages == {
        "benchmarks": 1.0,
        "method_extraction": 1.0,
        "readiness": 1.0,
        "report_coverage": 1.0,
    }


def test_run_deterministic_evaluation_tracks_partial_workspace_score():
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]

    summary = run_deterministic_evaluation([record], [_workspace(record, complete=False)])

    assert not summary.passed
    assert summary.missing_workspaces == []
    assert summary.check_averages["benchmarks"] == 0.25
    assert 0 < summary.average_score < 1


def test_summarize_evaluation_includes_missing_and_check_averages():
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:2]
    summary = run_deterministic_evaluation(records, [_workspace(records[0])])

    text = summarize_evaluation(summary)

    assert "records: 2" in text
    assert "matched_workspaces: 1" in text
    assert "missing: 2005.11401" in text
    assert "benchmarks: 1.000" in text
    assert "passed: false" in text


def test_load_workspace_records_reads_jsonl(tmp_path):
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]
    workspace = _workspace(record)
    path = tmp_path / "workspaces.jsonl"
    path.write_text(json.dumps(workspace.model_dump(mode="json")) + "\n", encoding="utf-8")

    loaded = load_workspace_records(path)

    assert len(loaded) == 1
    assert loaded[0].paper_id == "1706.03762"


def test_load_workspace_records_rejects_duplicate_paper_ids(tmp_path):
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]
    workspace = _workspace(record).model_dump(mode="json")
    path = tmp_path / "workspaces.jsonl"
    path.write_text(
        json.dumps(workspace) + "\n" + json.dumps(workspace) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunnerError, match="Duplicate workspace"):
        run_deterministic_evaluation([record], load_workspace_records(path))

