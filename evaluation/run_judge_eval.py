from __future__ import annotations

import argparse
import json

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records
from evaluation.judge_provider import ConfiguredLLMJudgeProvider, DryRunJudgeProvider
from evaluation.judge_rubrics import RubricLoaderError, load_judge_rubrics
from evaluation.judge_runner import build_judge_report
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
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured LLM provider and score judge tasks.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional judge model override for live mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Maximum output tokens per live judge call.",
    )
    args = parser.parse_args()

    if args.dry_run == args.live:
        print("ERROR choose exactly one of --dry-run or --live")
        return 1

    try:
        records = load_golden_records(args.golden)
        workspaces = load_workspace_records(args.workspaces)
        rubrics = load_judge_rubrics(args.rubrics)
        if args.live:
            provider = ConfiguredLLMJudgeProvider(
                requested_model=args.model,
                max_tokens=args.max_tokens,
            )
            mode = "live"
        else:
            provider = DryRunJudgeProvider()
            mode = "dry_run"
        report = build_judge_report(
            records=records,
            workspaces=workspaces,
            rubrics=rubrics,
            provider=provider,
            mode=mode,
        )
    except (GoldenDatasetError, EvaluationRunnerError, RubricLoaderError) as exc:
        print(f"ERROR {exc}")
        return 1

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
