from __future__ import annotations

import argparse
import json

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records
from evaluation.judge_rubrics import RubricLoaderError, load_judge_rubrics
from evaluation.judge_runner import build_dry_run_judge_report
from evaluation.runner import EvaluationRunnerError, load_workspace_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PaperIntel LLM-judge evaluation tasks."
    )
    parser.add_argument(
        "--golden",
        default="golden_dataset/seed_5.jsonl",
        help="Path to golden dataset JSONL.",
    )
    parser.add_argument(
        "--workspaces",
        required=True,
        help="Path to PaperWorkspace JSONL export.",
    )
    parser.add_argument(
        "--rubrics",
        default="evaluation/rubrics",
        help="Path to judge rubric directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build judge tasks without calling an LLM judge.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        print("ERROR only --dry-run mode is implemented")
        return 1

    try:
        records = load_golden_records(args.golden)
        workspaces = load_workspace_records(args.workspaces)
        rubrics = load_judge_rubrics(args.rubrics)
        report = build_dry_run_judge_report(records, workspaces, rubrics)
    except (GoldenDatasetError, EvaluationRunnerError, RubricLoaderError) as exc:
        print(f"ERROR {exc}")
        return 1

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

