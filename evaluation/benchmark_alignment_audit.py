from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmark_eval import (
    MATCHED,
    score_benchmark_candidate,
)
from evaluation.golden_dataset import (
    GoldenDatasetError,
    GoldenBenchmarkV01,
    GoldenDatasetRecord,
    GoldenDatasetRecordV02,
    load_golden_records,
)
from evaluation.runner import EvaluationRunnerError, load_workspace_records
from models.artifacts import PaperWorkspace


FAILURE_MATCHED = "matched"
FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET = (
    "right_value_metric_wrong_task_dataset"
)
FAILURE_WRONG_METHOD_VARIANT = "wrong_method_variant"
FAILURE_RIGHT_TASK_DATASET_WRONG_METRIC = "right_task_dataset_wrong_metric"
FAILURE_VALUE_MISMATCH = "value_mismatch"
FAILURE_UNIT_MISMATCH = "unit_mismatch"
FAILURE_CONDITION_MISMATCH = "condition_mismatch"
FAILURE_MISSING_DUE_TO_EXTRACTION = "missing_due_to_extraction"
FAILURE_EVALUATOR_NORMALIZATION_GAP = "evaluator_normalization_gap"
FAILURE_GOLDEN_LABEL_QUESTIONABLE = "golden_label_questionable"
FAILURE_WRONG_ROW_SELECTED = "wrong_row_selected"


@dataclass(frozen=True)
class BenchmarkAlignmentAudit:
    summary: dict[str, Any]
    papers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "papers": self.papers,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit expected benchmark rows against exported workspace rows."
    )
    parser.add_argument("--golden", required=True, help="Golden dataset JSONL path.")
    parser.add_argument("--workspaces", required=True, help="Workspace JSONL path.")
    parser.add_argument("--paper-ids", help="Comma-separated paper IDs to audit.")
    parser.add_argument("--top-k", type=int, default=5, help="Actual candidates per row.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be >= 1")
        records = load_golden_records(args.golden)
        paper_ids = _parse_csv(args.paper_ids)
        if paper_ids:
            records = [record for record in records if record.paper_id in paper_ids]
        if not records:
            raise ValueError("filters selected zero records")
        workspaces = load_workspace_records(args.workspaces)
        audit = run_benchmark_alignment_audit(
            records=records,
            workspaces=workspaces,
            top_k=args.top_k,
        )
    except (GoldenDatasetError, EvaluationRunnerError, ValueError) as exc:
        print(f"ERROR {exc}")
        return 1
    payload = audit.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(summarize_benchmark_alignment_audit(audit))
    return 0


def run_benchmark_alignment_audit(
    *,
    records: list[GoldenDatasetRecord],
    workspaces: list[PaperWorkspace],
    top_k: int = 5,
) -> BenchmarkAlignmentAudit:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    workspace_by_paper_id = {workspace.paper_id: workspace for workspace in workspaces}
    papers = [
        _audit_paper(record, workspace_by_paper_id.get(record.paper_id), top_k=top_k)
        for record in records
    ]
    return BenchmarkAlignmentAudit(
        summary=_global_summary(records, papers),
        papers=papers,
    )


def summarize_benchmark_alignment_audit(audit: BenchmarkAlignmentAudit) -> str:
    summary = audit.summary
    lines = [
        "PaperIntel benchmark alignment audit",
        f"papers: {summary['paper_count']}",
        f"total_expected: {summary['total_expected']}",
        f"matched_rows: {summary['matched_rows']}",
        f"failure_counts: {json.dumps(summary['failure_counts'], sort_keys=True)}",
    ]
    return "\n".join(lines)


def _audit_paper(
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace | None,
    *,
    top_k: int,
) -> dict[str, Any]:
    actual_rows = workspace.benchmarks_json if workspace is not None else []
    rows = [
        _audit_expected_row(record, expected, actual_rows, top_k=top_k)
        for expected in record.expected_benchmarks
    ]
    class_counts = Counter(row["primary_failure_class"] for row in rows)
    failure_counts = _failure_counts(class_counts)
    dominant_failure = failure_counts.most_common(1)[0][0] if failure_counts else None
    return {
        "paper_id": record.paper_id,
        "paper_family": record.paper_family if isinstance(record, GoldenDatasetRecordV02) else None,
        "difficulty_tags": record.difficulty_tags if isinstance(record, GoldenDatasetRecordV02) else [],
        "label_quality": record.label_quality,
        "workspace_found": workspace is not None,
        "expected_rows": len(record.expected_benchmarks),
        "actual_rows": len(actual_rows),
        "matched_rows": class_counts.get(FAILURE_MATCHED, 0),
        "class_counts": dict(sorted(class_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "dominant_failure": dominant_failure,
        "rows": rows,
    }


def _audit_expected_row(
    record: GoldenDatasetRecord,
    expected: GoldenBenchmarkV01,
    actual_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    candidates = _rank_candidates(expected, actual_rows, top_k=top_k)
    primary = candidates[0] if candidates else None
    primary_failure = (
        primary["failure_class"] if primary else FAILURE_MISSING_DUE_TO_EXTRACTION
    )
    if primary and primary["failure_class"] == FAILURE_MATCHED:
        primary_failure = FAILURE_MATCHED
    return {
        "expected": _expected_payload(expected),
        "expected_source": _expected_source(expected),
        "actual_candidates": candidates,
        "primary_failure_class": primary_failure,
    }


def _rank_candidates(
    expected: GoldenBenchmarkV01,
    actual_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    scored = []
    for actual in actual_rows:
        candidate_score = score_benchmark_candidate(expected, actual)
        coverage = _condition_keyword_coverage(expected, actual)
        failure_class = _classify_candidate(
            expected=expected,
            actual=actual,
            component_scores=candidate_score.component_scores,
            condition_keyword_coverage=coverage,
        )
        scored.append(
            {
                "row": actual,
                "actual_source": _actual_source(actual),
                "component_scores": candidate_score.component_scores,
                "score": candidate_score.score,
                "condition_keyword_coverage": coverage,
                "why_selected": _why_selected(candidate_score.component_scores),
                "failure_class": failure_class,
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: (
            item["component_scores"]["value"],
            item["component_scores"]["metric"],
            item["component_scores"]["task"],
            item["component_scores"]["dataset"],
            item["component_scores"]["conditions"],
            item["score"],
        ),
        reverse=True,
    )
    return [
        {
            "rank": index,
            **candidate,
        }
        for index, candidate in enumerate(ranked[:top_k], start=1)
    ]


def _classify_candidate(
    *,
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
    component_scores: dict[str, float],
    condition_keyword_coverage: dict[str, list[str]],
) -> str:
    if all(score == 1.0 for score in component_scores.values()):
        return FAILURE_MATCHED

    source_class = _source_failure_class(expected, actual)
    if source_class is not None:
        return source_class

    if _condition_suggests_wrong_method(expected, condition_keyword_coverage):
        return FAILURE_WRONG_METHOD_VARIANT

    if (
        component_scores["value"] == 1.0
        and component_scores["metric"] == 1.0
        and (
            component_scores["task"] < 1.0
            or component_scores["dataset"] < 1.0
        )
    ):
        return FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET

    if (
        component_scores["task"] == 1.0
        and component_scores["dataset"] == 1.0
        and component_scores["metric"] < 1.0
    ):
        return FAILURE_RIGHT_TASK_DATASET_WRONG_METRIC

    if (
        component_scores["task"] == 1.0
        and component_scores["metric"] == 1.0
        and component_scores["value"] < 1.0
    ):
        return FAILURE_VALUE_MISMATCH

    if (
        component_scores["task"] == 1.0
        and component_scores["metric"] == 1.0
        and component_scores["value"] == 1.0
        and component_scores["unit"] < 1.0
    ):
        return FAILURE_UNIT_MISMATCH

    if (
        component_scores["task"] == 1.0
        and component_scores["metric"] == 1.0
        and component_scores["value"] == 1.0
        and component_scores["unit"] == 1.0
        and component_scores["conditions"] < 1.0
    ):
        return FAILURE_CONDITION_MISMATCH

    if _looks_like_normalization_gap(expected, actual, component_scores):
        return FAILURE_EVALUATOR_NORMALIZATION_GAP

    return FAILURE_WRONG_ROW_SELECTED


def _source_failure_class(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> str | None:
    expected_table = getattr(expected, "source_table_or_figure", None)
    actual_table = actual.get("source_table_or_figure")
    if expected_table and actual_table and str(expected_table) != str(actual_table):
        return FAILURE_WRONG_ROW_SELECTED
    return None


def _condition_suggests_wrong_method(
    expected: GoldenBenchmarkV01,
    coverage: dict[str, list[str]],
) -> bool:
    if not expected.conditions_keywords:
        return False
    missing = coverage["missing"]
    return bool(missing) and len(missing) == len(expected.conditions_keywords)


def _looks_like_normalization_gap(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
    component_scores: dict[str, float],
) -> bool:
    if component_scores["value"] != 1.0:
        return False
    expected_terms = {
        str(getattr(expected, "task", "")).casefold(),
        str(getattr(expected, "dataset", "")).casefold(),
    }
    actual_terms = {
        str(actual.get("task", "")).casefold(),
        str(actual.get("dataset", "")).casefold(),
    }
    return bool(expected_terms.intersection(actual_terms))


def _condition_keyword_coverage(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> dict[str, list[str]]:
    search_text = " ".join(
        str(value or "").casefold()
        for value in [
            actual.get("task"),
            actual.get("dataset"),
            actual.get("metric"),
            actual.get("conditions"),
            " ".join(str(item) for item in actual.get("conditions_keywords", []))
            if isinstance(actual.get("conditions_keywords"), list)
            else actual.get("conditions_keywords"),
        ]
    )
    matched = [
        keyword
        for keyword in expected.conditions_keywords
        if str(keyword).casefold() in search_text
    ]
    missing = [
        keyword
        for keyword in expected.conditions_keywords
        if keyword not in matched
    ]
    return {
        "matched": matched,
        "missing": missing,
    }


def _why_selected(component_scores: dict[str, float]) -> list[str]:
    labels = []
    for component, score in component_scores.items():
        if score == 1.0:
            labels.append(f"{component}_match")
        elif score > 0.0:
            labels.append(f"{component}_partial")
    return labels


def _expected_payload(expected: GoldenBenchmarkV01) -> dict[str, Any]:
    return {
        "task": expected.task,
        "dataset": getattr(expected, "dataset", None),
        "metric": expected.metric,
        "value": expected.value,
        "unit": expected.unit,
        "conditions_keywords": expected.conditions_keywords,
        "reported_as": getattr(expected, "reported_as", None),
        "value_type": getattr(expected, "value_type", None),
    }


def _expected_source(expected: GoldenBenchmarkV01) -> dict[str, Any]:
    anchor = getattr(expected, "evidence_anchor", None)
    return {
        "source_section": getattr(expected, "source_section", None),
        "source_table_or_figure": getattr(expected, "source_table_or_figure", None),
        "evidence_anchor": anchor.model_dump() if anchor else None,
    }


def _actual_source(actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_section": actual.get("source_section"),
        "source_table_or_figure": actual.get("source_table_or_figure"),
        "evidence_anchor": actual.get("evidence_anchor"),
    }


def _global_summary(
    records: list[GoldenDatasetRecord],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_counts = Counter()
    by_paper_family: dict[str, Counter] = defaultdict(Counter)
    by_difficulty_tag: dict[str, Counter] = defaultdict(Counter)
    normalization_gaps = Counter()
    matched_rows = 0
    total_expected = 0

    for paper in papers:
        total_expected += paper["expected_rows"]
        matched_rows += paper["matched_rows"]
        for failure_class, count in paper["failure_counts"].items():
            failure_counts[failure_class] += count
            family = paper.get("paper_family")
            if family:
                by_paper_family[family][failure_class] += count
            for tag in paper.get("difficulty_tags") or []:
                by_difficulty_tag[tag][failure_class] += count
        for row in paper["rows"]:
            primary = row["actual_candidates"][0] if row["actual_candidates"] else None
            if not primary:
                continue
            if primary["failure_class"] in {
                FAILURE_RIGHT_VALUE_METRIC_WRONG_TASK_DATASET,
                FAILURE_EVALUATOR_NORMALIZATION_GAP,
            }:
                expected = row["expected"]
                actual = primary["row"]
                normalization_gaps[
                    (
                        str(expected.get("task")),
                        str(expected.get("dataset")),
                        str(actual.get("task")),
                        str(actual.get("dataset")),
                    )
                ] += 1

    return {
        "paper_count": len(records),
        "total_expected": total_expected,
        "matched_rows": matched_rows,
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_paper_family": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(by_paper_family.items())
        },
        "by_difficulty_tag": {
            tag: dict(sorted(counter.items()))
            for tag, counter in sorted(by_difficulty_tag.items())
        },
        "top_normalization_gaps": [
            {
                "expected_task": key[0],
                "expected_dataset": key[1],
                "actual_task": key[2],
                "actual_dataset": key[3],
                "count": count,
            }
            for key, count in normalization_gaps.most_common(10)
        ],
    }


def _failure_counts(class_counts: Counter) -> Counter:
    return Counter(
        {
            class_name: count
            for class_name, count in class_counts.items()
            if class_name != FAILURE_MATCHED
        }
    )


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
