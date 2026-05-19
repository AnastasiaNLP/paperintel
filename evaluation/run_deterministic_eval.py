from __future__ import annotations

import argparse
import json

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records
from evaluation.runner import (
    EvaluationRunnerError,
    load_workspace_records,
    run_deterministic_evaluation,
    summarize_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic PaperIntel eval over exported workspaces."
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
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )
    args = parser.parse_args()

    try:
        records = load_golden_records(args.golden)
        workspaces = load_workspace_records(args.workspaces)
        summary = run_deterministic_evaluation(records, workspaces)
    except (GoldenDatasetError, EvaluationRunnerError) as exc:
        print(f"ERROR {exc}")
        return 1

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print(summarize_evaluation(summary))
    return 0 if summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

