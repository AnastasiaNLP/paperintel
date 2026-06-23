from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmark_eval import evaluate_benchmarks
from evaluation.golden_dataset import (
    GoldenDatasetError,
    GoldenDatasetRecord,
    GoldenDatasetRecordV02,
    build_golden_summary,
    load_golden_records,
)
from evaluation.method_eval import evaluate_method
from evaluation.runner import EvaluationRunnerError, load_workspace_records
from models.artifacts import PaperWorkspace


EVAL_FRAMEWORK_VERSION = "paperintel_eval_v1"
SUPPORTED_CHECKS = {"method", "benchmark"}


@dataclass(frozen=True)
class EvalV1Result:
    paper_id: str
    label_quality: str
    paper_family: str | None
    difficulty_tags: list[str]
    checks: dict[str, dict[str, Any]]
    score: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "label_quality": self.label_quality,
            "paper_family": self.paper_family,
            "difficulty_tags": self.difficulty_tags,
            "checks": self.checks,
            "score": self.score,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EvalV1Summary:
    provenance: dict[str, Any]
    total_records: int
    selected_record_count: int
    matched_workspaces: int
    missing_workspaces: list[str]
    average_score: float
    passed: bool
    breakdowns: dict[str, dict[str, Any]]
    results: list[EvalV1Result]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "total_records": self.total_records,
            "selected_record_count": self.selected_record_count,
            "matched_workspaces": self.matched_workspaces,
            "missing_workspaces": self.missing_workspaces,
            "average_score": self.average_score,
            "passed": self.passed,
            "breakdowns": self.breakdowns,
            "results": [result.to_dict() for result in self.results],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PaperIntel eval v1 over exported workspaces."
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
        "--checks",
        default="method,benchmark",
        help="Comma-separated checks to run: method,benchmark.",
    )
    parser.add_argument("--paper-ids", help="Comma-separated paper IDs to evaluate.")
    parser.add_argument(
        "--label-quality",
        help="Comma-separated label_quality values to include.",
    )
    parser.add_argument(
        "--paper-family",
        help="Comma-separated v0.2 paper_family values to include.",
    )
    parser.add_argument(
        "--difficulty-tags",
        help="Comma-separated v0.2 difficulty tags; records matching any tag are included.",
    )
    parser.add_argument("--summary-output", help="Optional path for JSON summary output.")
    parser.add_argument("--results-output", help="Optional path for JSONL paper results.")
    args = parser.parse_args()

    try:
        checks = _parse_checks(args.checks)
        records = load_golden_records(args.golden)
        selected_records = _filter_records(
            records,
            paper_ids=_parse_csv(args.paper_ids),
            label_quality=_parse_csv(args.label_quality),
            paper_family=_parse_csv(args.paper_family),
            difficulty_tags=_parse_csv(args.difficulty_tags),
        )
        if not selected_records:
            raise ValueError("filters selected zero records")
        workspaces = load_workspace_records(args.workspaces)
        summary = run_eval_v1(
            records=records,
            selected_records=selected_records,
            workspaces=workspaces,
            checks=checks,
            dataset_path=args.golden,
            workspace_path=args.workspaces,
        )
        if args.summary_output:
            _write_json(Path(args.summary_output), summary.to_dict())
        if args.results_output:
            _write_jsonl(
                Path(args.results_output),
                [result.to_dict() for result in summary.results],
            )
    except (GoldenDatasetError, EvaluationRunnerError, ValueError) as exc:
        print(f"ERROR {exc}")
        return 1

    print(summarize_eval_v1(summary))
    return 0 if summary.passed else 2


def run_eval_v1(
    *,
    records: list[GoldenDatasetRecord],
    selected_records: list[GoldenDatasetRecord],
    workspaces: list[PaperWorkspace],
    checks: list[str],
    dataset_path: str,
    workspace_path: str,
) -> EvalV1Summary:
    workspace_by_paper_id = _index_workspaces(workspaces)
    dataset_summary = build_golden_summary(records)
    selected_summary = build_golden_summary(selected_records) if selected_records else None
    results: list[EvalV1Result] = []
    missing_workspaces: list[str] = []

    for record in selected_records:
        workspace = workspace_by_paper_id.get(record.paper_id)
        if workspace is None:
            missing_workspaces.append(record.paper_id)
            continue
        results.append(_evaluate_record(record, workspace, checks))

    average_score = _average([result.score for result in results])
    passed = not missing_workspaces and all(result.passed for result in results)
    provenance = {
        "eval_framework_version": EVAL_FRAMEWORK_VERSION,
        "dataset_path": dataset_path,
        "workspace_path": workspace_path,
        "dataset_versions": dataset_summary.dataset_versions,
        "schema_versions": dataset_summary.schema_versions,
        "selected_dataset_versions": selected_summary.dataset_versions if selected_summary else {},
        "selected_schema_versions": selected_summary.schema_versions if selected_summary else {},
        "checks": checks,
        "workspace_count": len(workspaces),
        "selected_record_count": len(selected_records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return EvalV1Summary(
        provenance=provenance,
        total_records=len(records),
        selected_record_count=len(selected_records),
        matched_workspaces=len(results),
        missing_workspaces=missing_workspaces,
        average_score=average_score,
        passed=passed,
        breakdowns=_build_breakdowns(selected_records, results),
        results=results,
    )


def _evaluate_record(
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace,
    checks: list[str],
) -> EvalV1Result:
    check_results: dict[str, dict[str, Any]] = {}
    scores: list[float] = []
    for check in checks:
        if check == "method":
            result = evaluate_method(record, workspace)
        elif check == "benchmark":
            result = evaluate_benchmarks(record, workspace)
        else:
            raise ValueError(f"Unsupported check: {check}")
        check_results[check] = result.to_dict()
        scores.append(result.score)

    score = _average(scores)
    return EvalV1Result(
        paper_id=record.paper_id,
        label_quality=record.label_quality,
        paper_family=record.paper_family if isinstance(record, GoldenDatasetRecordV02) else None,
        difficulty_tags=record.difficulty_tags if isinstance(record, GoldenDatasetRecordV02) else [],
        checks=check_results,
        score=score,
        passed=all(check["passed"] for check in check_results.values()),
    )


def _filter_records(
    records: list[GoldenDatasetRecord],
    *,
    paper_ids: set[str],
    label_quality: set[str],
    paper_family: set[str],
    difficulty_tags: set[str],
) -> list[GoldenDatasetRecord]:
    return [
        record
        for record in records
        if (not paper_ids or record.paper_id in paper_ids)
        and (not label_quality or record.label_quality in label_quality)
        and (
            not paper_family
            or (
                isinstance(record, GoldenDatasetRecordV02)
                and record.paper_family in paper_family
            )
        )
        and (
            not difficulty_tags
            or (
                isinstance(record, GoldenDatasetRecordV02)
                and bool(set(record.difficulty_tags) & difficulty_tags)
            )
        )
    ]


def _build_breakdowns(
    selected_records: list[GoldenDatasetRecord],
    results: list[EvalV1Result],
) -> dict[str, dict[str, Any]]:
    result_by_paper_id = {result.paper_id: result for result in results}
    grouped_scores: dict[str, dict[str, list[float]]] = {
        "label_quality": defaultdict(list),
        "paper_family": defaultdict(list),
        "difficulty_tags": defaultdict(list),
    }
    counts: dict[str, Counter[str]] = {
        "label_quality": Counter(),
        "paper_family": Counter(),
        "difficulty_tags": Counter(),
    }
    for record in selected_records:
        result = result_by_paper_id.get(record.paper_id)
        counts["label_quality"][record.label_quality] += 1
        if result is not None:
            grouped_scores["label_quality"][record.label_quality].append(result.score)
        if isinstance(record, GoldenDatasetRecordV02):
            counts["paper_family"][record.paper_family] += 1
            if result is not None:
                grouped_scores["paper_family"][record.paper_family].append(result.score)
            for tag in record.difficulty_tags:
                counts["difficulty_tags"][tag] += 1
                if result is not None:
                    grouped_scores["difficulty_tags"][tag].append(result.score)
    return {
        name: {
            "counts": dict(sorted(count.items())),
            "average_scores": {
                key: _average(scores)
                for key, scores in sorted(grouped_scores[name].items())
            },
        }
        for name, count in counts.items()
    }


def _parse_checks(raw: str) -> list[str]:
    checks = _parse_csv_ordered(raw)
    if not checks:
        raise ValueError("At least one check must be provided.")
    unsupported = set(checks) - SUPPORTED_CHECKS
    if unsupported:
        raise ValueError(f"Unsupported checks: {','.join(sorted(unsupported))}")
    return checks


def _parse_csv_ordered(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    values: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _parse_csv(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def _index_workspaces(workspaces: list[PaperWorkspace]) -> dict[str, PaperWorkspace]:
    indexed: dict[str, PaperWorkspace] = {}
    duplicates: list[str] = []
    for workspace in workspaces:
        if workspace.paper_id in indexed:
            duplicates.append(workspace.paper_id)
        indexed[workspace.paper_id] = workspace
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate workspace paper_id values: {duplicate_list}")
    return indexed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize_eval_v1(summary: EvalV1Summary) -> str:
    return "\n".join(
        [
            "PaperIntel eval v1",
            f"records: {summary.selected_record_count}",
            f"matched_workspaces: {summary.matched_workspaces}",
            f"missing_workspaces: {len(summary.missing_workspaces)}",
            "checks: " + ",".join(summary.provenance["checks"]),
            f"average_score: {summary.average_score:.3f}",
            f"passed: {str(summary.passed).lower()}",
        ]
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
