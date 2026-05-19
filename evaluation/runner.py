from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evaluation.deterministic_metrics import WorkspaceEvaluation, evaluate_workspace
from evaluation.golden_dataset import GoldenDatasetError, GoldenDatasetRecord
from models.artifacts import PaperWorkspace


class EvaluationRunnerError(ValueError):
    """Raised when deterministic evaluation inputs are invalid."""


@dataclass(frozen=True)
class DeterministicEvaluationSummary:
    total_records: int
    matched_workspaces: int
    missing_workspaces: list[str]
    average_score: float
    check_averages: dict[str, float]
    paper_results: list[WorkspaceEvaluation]

    @property
    def passed(self) -> bool:
        return not self.missing_workspaces and all(
            result.passed for result in self.paper_results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "matched_workspaces": self.matched_workspaces,
            "missing_workspaces": self.missing_workspaces,
            "average_score": self.average_score,
            "check_averages": self.check_averages,
            "passed": self.passed,
            "paper_results": [
                {
                    "paper_id": result.paper_id,
                    "score": result.score,
                    "passed": result.passed,
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "score": check.score,
                            "details": check.details,
                        }
                        for check in result.checks
                    ],
                }
                for result in self.paper_results
            ],
        }


def load_workspace_records(path: str | Path) -> list[PaperWorkspace]:
    workspace_path = Path(path)
    try:
        lines = workspace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationRunnerError(f"Could not read workspaces: {workspace_path}") from exc

    workspaces: list[PaperWorkspace] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        workspaces.append(
            _parse_workspace(line, line_number=line_number, path=workspace_path)
        )

    if not workspaces:
        raise EvaluationRunnerError(f"Workspace JSONL is empty: {workspace_path}")

    return workspaces


def run_deterministic_evaluation(
    records: list[GoldenDatasetRecord],
    workspaces: list[PaperWorkspace],
) -> DeterministicEvaluationSummary:
    workspace_by_paper_id = _index_workspaces(workspaces)
    paper_results: list[WorkspaceEvaluation] = []
    missing_workspaces: list[str] = []

    for record in records:
        workspace = workspace_by_paper_id.get(record.paper_id)
        if workspace is None:
            missing_workspaces.append(record.paper_id)
            continue
        paper_results.append(evaluate_workspace(record, workspace))

    return DeterministicEvaluationSummary(
        total_records=len(records),
        matched_workspaces=len(paper_results),
        missing_workspaces=missing_workspaces,
        average_score=_average([result.score for result in paper_results]),
        check_averages=_check_averages(paper_results),
        paper_results=paper_results,
    )


def summarize_evaluation(summary: DeterministicEvaluationSummary) -> str:
    lines = [
        "Deterministic evaluation",
        f"records: {summary.total_records}",
        f"matched_workspaces: {summary.matched_workspaces}",
        f"missing_workspaces: {len(summary.missing_workspaces)}",
        f"average_score: {summary.average_score:.3f}",
        f"passed: {str(summary.passed).lower()}",
    ]
    if summary.missing_workspaces:
        lines.append("missing: " + ",".join(summary.missing_workspaces))
    for name, score in sorted(summary.check_averages.items()):
        lines.append(f"{name}: {score:.3f}")
    return "\n".join(lines)


def _parse_workspace(
    line: str,
    *,
    line_number: int,
    path: Path,
) -> PaperWorkspace:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EvaluationRunnerError(
            f"Invalid workspace JSON in {path} at line {line_number}: {exc.msg}"
        ) from exc

    try:
        return PaperWorkspace.model_validate(payload)
    except ValidationError as exc:
        raise EvaluationRunnerError(
            f"Invalid workspace record in {path} at line {line_number}: {exc}"
        ) from exc


def _index_workspaces(workspaces: list[PaperWorkspace]) -> dict[str, PaperWorkspace]:
    indexed: dict[str, PaperWorkspace] = {}
    duplicates: list[str] = []
    for workspace in workspaces:
        if workspace.paper_id in indexed:
            duplicates.append(workspace.paper_id)
        indexed[workspace.paper_id] = workspace

    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise EvaluationRunnerError(f"Duplicate workspace paper_id values: {duplicate_list}")

    return indexed


def _check_averages(
    paper_results: list[WorkspaceEvaluation],
) -> dict[str, float]:
    scores_by_name: dict[str, list[float]] = {}
    for result in paper_results:
        for check in result.checks:
            scores_by_name.setdefault(check.name, []).append(check.score)
    return {
        name: _average(scores)
        for name, scores in sorted(scores_by_name.items())
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

