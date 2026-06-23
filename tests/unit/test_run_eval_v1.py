import json
import subprocess
import sys

from evaluation.golden_dataset import load_golden_records
from evaluation.run_eval import _filter_records


GOLDEN_PATH = "golden_dataset/seed_5.jsonl"
GOLDEN_V02_PATH = "golden_dataset/paperintel_60_v2_2.jsonl"
WORKSPACES_PATH = "tests/fixtures/evaluation/workspaces_seed_sample.jsonl"


def _run_eval(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "evaluation.run_eval", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_run_eval_v1_method_only_text_contract():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "1706.03762,2005.11401",
    )

    assert result.returncode == 0
    assert "PaperIntel eval v1" in result.stdout
    assert "records: 2" in result.stdout
    assert "matched_workspaces: 2" in result.stdout
    assert "missing_workspaces: 0" in result.stdout
    assert "checks: method" in result.stdout
    assert "passed: true" in result.stdout


def test_run_eval_v1_method_and_benchmark_returns_failed_exit_for_partial_fixture():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method,benchmark",
        "--paper-ids",
        "1706.03762,2005.11401",
    )

    assert result.returncode == 2
    assert "records: 2" in result.stdout
    assert "matched_workspaces: 2" in result.stdout
    assert "checks: method,benchmark" in result.stdout
    assert "passed: false" in result.stdout


def test_run_eval_v1_paper_id_filter_limits_records():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "1706.03762",
    )

    assert result.returncode == 0
    assert "records: 1" in result.stdout
    assert "matched_workspaces: 1" in result.stdout


def test_run_eval_v1_label_quality_filter():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--label-quality",
        "manual_verified",
        "--paper-ids",
        "1706.03762",
    )

    assert result.returncode == 0
    assert "records: 1" in result.stdout


def test_run_eval_v1_unknown_check_exits_1():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method,unknown",
    )

    assert result.returncode == 1
    assert "Unsupported checks: unknown" in result.stdout


def test_run_eval_v1_zero_selected_records_exits_1():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "does-not-exist",
    )

    assert result.returncode == 1
    assert "filters selected zero records" in result.stdout
    assert "PaperIntel eval v1" not in result.stdout


def test_run_eval_v1_missing_workspace_exits_2():
    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "2106.09685",
    )

    assert result.returncode == 2
    assert "records: 1" in result.stdout
    assert "matched_workspaces: 0" in result.stdout
    assert "missing_workspaces: 1" in result.stdout
    assert "passed: false" in result.stdout


def test_run_eval_v1_writes_summary_and_results(tmp_path):
    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.jsonl"

    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "1706.03762,2005.11401",
        "--summary-output",
        str(summary_path),
        "--results-output",
        str(results_path),
    )

    assert result.returncode == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["provenance"]["eval_framework_version"] == "paperintel_eval_v1"
    assert summary["provenance"]["dataset_path"] == GOLDEN_PATH
    assert summary["provenance"]["workspace_path"] == WORKSPACES_PATH
    assert summary["provenance"]["checks"] == ["method"]
    assert summary["provenance"]["workspace_count"] == 2
    assert summary["provenance"]["selected_record_count"] == 2
    assert summary["selected_record_count"] == 2
    assert len(result_rows) == 2
    assert result_rows[0]["checks"]["method"]["passed"] is True


def test_run_eval_v1_summary_breakdowns_are_present(tmp_path):
    summary_path = tmp_path / "summary.json"

    result = _run_eval(
        "--golden",
        GOLDEN_PATH,
        "--workspaces",
        WORKSPACES_PATH,
        "--checks",
        "method",
        "--paper-ids",
        "1706.03762,2005.11401",
        "--summary-output",
        str(summary_path),
    )

    assert result.returncode == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["breakdowns"]["label_quality"]["counts"] == {
        "manual_verified": 2
    }
    assert summary["breakdowns"]["label_quality"]["average_scores"] == {
        "manual_verified": 1.0
    }


def test_run_eval_v1_v02_filters_select_records():
    records = load_golden_records(GOLDEN_V02_PATH)

    selected = _filter_records(
        records,
        paper_ids=set(),
        label_quality={"manual_verified"},
        paper_family={"architecture"},
        difficulty_tags={"table_heavy"},
    )

    assert selected
    assert all(record.label_quality == "manual_verified" for record in selected)
    assert all(record.paper_family == "architecture" for record in selected)
    assert all("table_heavy" in record.difficulty_tags for record in selected)
