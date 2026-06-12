from __future__ import annotations

import argparse
import json

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records
from evaluation.judge_automation import (
    JudgeAutomationError,
    compare_judge_results,
    load_judge_results_jsonl,
    write_baseline_comparison,
    write_judge_results_jsonl,
    write_judge_summary,
)
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
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for JSONL JudgeResult records.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for the JudgeRunReport JSON summary.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline JSONL JudgeResult file to compare against.",
    )
    parser.add_argument(
        "--compare-output",
        default=None,
        help="Optional path for baseline comparison JSON.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute score delta for improved/regressed classification.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional dataset version label stored on judge tasks.",
    )
    parser.add_argument(
        "--pipeline-version",
        default=None,
        help="Optional pipeline version label stored on judge tasks.",
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
            judge_model = args.model or "configured"
            provider = ConfiguredLLMJudgeProvider(
                requested_model=args.model,
                max_tokens=args.max_tokens,
            )
            mode = "live"
        else:
            judge_model = "dry_run"
            provider = DryRunJudgeProvider()
            mode = "dry_run"
        report = build_judge_report(
            records=records,
            workspaces=workspaces,
            rubrics=rubrics,
            provider=provider,
            mode=mode,
            judge_model=judge_model,
            dataset_version=args.dataset_version,
            pipeline_version=args.pipeline_version,
        )
        if args.output:
            write_judge_results_jsonl(report.results, args.output)
        if args.summary_output:
            write_judge_summary(report, args.summary_output)
        if args.baseline:
            comparison = compare_judge_results(
                current=report.results,
                baseline=load_judge_results_jsonl(args.baseline),
                min_delta=args.min_delta,
            )
            if args.compare_output:
                write_baseline_comparison(comparison, args.compare_output)
            else:
                print(
                    json.dumps(
                        comparison.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
    except (
        GoldenDatasetError,
        EvaluationRunnerError,
        RubricLoaderError,
        JudgeAutomationError,
        ValueError,
    ) as exc:
        print(f"ERROR {exc}")
        return 1

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
