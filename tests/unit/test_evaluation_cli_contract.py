import json
import subprocess
import sys


GOLDEN_PATH = "golden_dataset/seed_5.jsonl"
WORKSPACES_PATH = "tests/fixtures/evaluation/workspaces_seed_sample.jsonl"


def test_validate_golden_dataset_cli_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.validate_golden_dataset",
            GOLDEN_PATH,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "OK records=5 paper_ids="
        "1706.03762,2005.11401,2106.09685,2210.03629,2205.14135"
    )


def test_run_deterministic_eval_text_cli_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_deterministic_eval",
            "--golden",
            GOLDEN_PATH,
            "--workspaces",
            WORKSPACES_PATH,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Deterministic evaluation" in result.stdout
    assert "records: 5" in result.stdout
    assert "matched_workspaces: 2" in result.stdout
    assert "missing_workspaces: 3" in result.stdout
    assert "missing: 2106.09685,2210.03629,2205.14135" in result.stdout
    assert "passed: false" in result.stdout


def test_run_deterministic_eval_json_cli_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_deterministic_eval",
            "--golden",
            GOLDEN_PATH,
            "--workspaces",
            WORKSPACES_PATH,
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["total_records"] == 5
    assert payload["matched_workspaces"] == 2
    assert payload["missing_workspaces"] == [
        "2106.09685",
        "2210.03629",
        "2205.14135",
    ]
    assert not payload["passed"]
    assert [paper["paper_id"] for paper in payload["paper_results"]] == [
        "1706.03762",
        "2005.11401",
    ]
    assert payload["paper_results"][0]["passed"] is True
    assert payload["paper_results"][1]["passed"] is False

