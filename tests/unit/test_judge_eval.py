import json
import subprocess
import sys

from evaluation.golden_dataset import load_golden_records
from evaluation.judge_rubrics import EXPECTED_RUBRIC_IDS, load_judge_rubrics
from evaluation.judge_runner import build_dry_run_judge_report
from evaluation.runner import load_workspace_records


WORKSPACES_PATH = "tests/fixtures/evaluation/workspaces_seed_sample.jsonl"


def test_load_judge_rubrics_finds_expected_versioned_files():
    rubrics = load_judge_rubrics()

    assert set(rubrics) == EXPECTED_RUBRIC_IDS
    assert all(len(rubric.sha256) == 64 for rubric in rubrics.values())
    assert rubrics["recommended_action"].text.startswith("# Rubric:")


def test_dry_run_judge_report_builds_report_tasks_for_matched_workspaces():
    records = load_golden_records("golden_dataset/seed_5.jsonl")
    workspaces = load_workspace_records(WORKSPACES_PATH)
    rubrics = load_judge_rubrics()

    report = build_dry_run_judge_report(records, workspaces, rubrics)

    assert report.mode == "dry_run"
    assert report.total_tasks == 6
    assert report.scored_tasks == 0
    assert {result.status for result in report.results} == {"not_scored"}
    assert {result.task.paper_id for result in report.results} == {
        "1706.03762",
        "2005.11401",
    }
    assert {result.task.rubric_id for result in report.results} == {
        "recommended_action",
        "implementation_difficulty",
        "action_reasoning",
    }
    assert all(result.task.rubric_hash for result in report.results)


def test_run_judge_eval_dry_run_cli_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_judge_eval",
            "--golden",
            "golden_dataset/seed_5.jsonl",
            "--workspaces",
            WORKSPACES_PATH,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["mode"] == "dry_run"
    assert payload["total_tasks"] == 6
    assert payload["scored_tasks"] == 0
    assert {item["status"] for item in payload["results"]} == {"not_scored"}
    assert all(item["task"]["rubric_hash"] for item in payload["results"])

