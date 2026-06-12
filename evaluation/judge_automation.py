from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from evaluation.judge_models import (
    JudgeBaselineComparison,
    JudgeBaselineDelta,
    JudgeResult,
    JudgeRunReport,
)


class JudgeAutomationError(ValueError):
    """Raised when automated judge result files cannot be read or written."""


def finalize_judge_report(report: JudgeRunReport) -> JudgeRunReport:
    status_counts = Counter(result.status for result in report.results)
    scored = [result for result in report.results if result.status == "scored"]
    scores_by_rubric: dict[str, list[float]] = defaultdict(list)
    for result in scored:
        if result.score is not None:
            scores_by_rubric[result.task.rubric_id].append(result.score)

    report.total_tasks = len(report.results)
    report.scored_tasks = len(scored)
    report.status_counts = dict(sorted(status_counts.items()))
    report.average_scores_by_rubric = {
        rubric_id: _average(scores)
        for rubric_id, scores in sorted(scores_by_rubric.items())
    }
    report.average_score = (
        _average([result.score for result in scored if result.score is not None])
        if scored
        else None
    )
    return report


def write_judge_results_jsonl(results: list[JudgeResult], path: str | Path) -> None:
    output_path = Path(path)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
    except OSError as exc:
        raise JudgeAutomationError(
            f"Could not write judge results: {output_path}"
        ) from exc


def write_judge_summary(report: JudgeRunReport, path: str | Path) -> None:
    output_path = Path(path)
    try:
        output_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise JudgeAutomationError(
            f"Could not write judge summary: {output_path}"
        ) from exc


def load_judge_results_jsonl(path: str | Path) -> list[JudgeResult]:
    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JudgeAutomationError(f"Could not read judge results: {input_path}") from exc

    results: list[JudgeResult] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JudgeAutomationError(
                "Invalid judge result JSON in "
                f"{input_path} at line {line_number}: {exc.msg}"
            ) from exc
        try:
            results.append(JudgeResult.model_validate(payload))
        except ValidationError as exc:
            raise JudgeAutomationError(
                f"Invalid judge result record in {input_path} at line {line_number}: {exc}"
            ) from exc

    if not results:
        raise JudgeAutomationError(f"Judge result JSONL is empty: {input_path}")
    return results


def compare_judge_results(
    *,
    current: list[JudgeResult],
    baseline: list[JudgeResult],
    min_delta: float = 0.0,
) -> JudgeBaselineComparison:
    if min_delta < 0:
        raise ValueError("min_delta must not be negative")

    current_index = _scored_index(current, label="current")
    baseline_index = _scored_index(baseline, label="baseline")
    current_keys = set(current_index)
    baseline_keys = set(baseline_index)
    matched_keys = sorted(current_keys & baseline_keys)

    comparison = JudgeBaselineComparison(
        current_count=len(current),
        baseline_count=len(baseline),
        matched_scored_tasks=len(matched_keys),
        missing_in_current=[
            _key_to_string(key) for key in sorted(baseline_keys - current_keys)
        ],
        missing_in_baseline=[
            _key_to_string(key) for key in sorted(current_keys - baseline_keys)
        ],
    )

    for key in matched_keys:
        current_result = current_index[key]
        baseline_result = baseline_index[key]
        assert current_result.score is not None
        assert baseline_result.score is not None
        delta = current_result.score - baseline_result.score
        item = JudgeBaselineDelta(
            task_family=key[0],
            sample_id=key[1],
            rubric_id=key[2],
            current_score=current_result.score,
            baseline_score=baseline_result.score,
            delta=delta,
        )
        if delta > min_delta:
            comparison.improved.append(item)
        elif delta < -min_delta:
            comparison.regressed.append(item)
        else:
            comparison.unchanged.append(item)
    return comparison


def write_baseline_comparison(
    comparison: JudgeBaselineComparison,
    path: str | Path,
) -> None:
    output_path = Path(path)
    try:
        output_path.write_text(
            json.dumps(comparison.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise JudgeAutomationError(
            f"Could not write judge baseline comparison: {output_path}"
        ) from exc


def _scored_index(
    results: list[JudgeResult],
    *,
    label: str,
) -> dict[tuple[str, str, str], JudgeResult]:
    indexed: dict[tuple[str, str, str], JudgeResult] = {}
    for result in results:
        if result.status != "scored" or result.score is None:
            continue
        key = (
            result.task.task_family,
            _result_sample_id(result),
            result.task.rubric_id,
        )
        if key in indexed:
            raise JudgeAutomationError(
                f"Duplicate scored judge result in {label}: {_key_to_string(key)}"
            )
        indexed[key] = result
    return indexed


def _result_sample_id(result: JudgeResult) -> str:
    return result.task.sample_id or result.task.paper_id


def _key_to_string(key: tuple[str, str, str]) -> str:
    return f"{key[0]}::{key[1]}::{key[2]}"


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
