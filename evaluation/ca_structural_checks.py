from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from models.artifacts import ComparisonArtifact
from models.schemas import ComparisonReport
from models.synthesis import SynthesisAgentResult, SynthesisReport


@dataclass(frozen=True)
class StructuralCheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def check_comparison_artifact(
    artifact: ComparisonArtifact,
    *,
    requested_paper_ids: list[str],
) -> StructuralCheckResult:
    errors: list[str] = []
    requested = list(dict.fromkeys(requested_paper_ids))
    if artifact.paper_ids != requested:
        errors.append("artifact.paper_ids does not match requested paper order.")

    try:
        report = ComparisonReport.model_validate(artifact.comparison_report_json)
    except ValidationError as exc:
        return StructuralCheckResult(
            passed=False,
            errors=[f"comparison_report_json is invalid: {exc}"],
        )

    if report.producer != "comparison_analyst":
        errors.append("comparison report producer is not comparison_analyst.")

    expected_indexes = set(range(len(requested)))
    summary_indexes = {
        item.get("paper_index")
        for item in report.papers_summary
        if isinstance(item.get("paper_index"), int)
    }
    missing_summary = expected_indexes - summary_indexes
    if missing_summary:
        errors.append(
            "papers_summary is missing requested paper indexes: "
            + ",".join(str(index) for index in sorted(missing_summary))
        )

    requested_set = set(requested)
    unknown_summary_ids = sorted(
        arxiv_id
        for item in report.papers_summary
        if (arxiv_id := item.get("arxiv_id")) and arxiv_id not in requested_set
    )
    if unknown_summary_ids:
        errors.append(
            "papers_summary contains unknown paper ids: "
            + ",".join(unknown_summary_ids)
        )

    represented_indexes: set[int] = set()
    for row in report.comparison_matrix:
        represented_indexes.update(
            index
            for index in row.values_by_paper
            if index in expected_indexes
        )
        unknown_indexes = set(row.values_by_paper) - expected_indexes
        if unknown_indexes:
            errors.append(
                "comparison_matrix contains unknown paper indexes: "
                + ",".join(str(index) for index in sorted(unknown_indexes))
            )

    missing_matrix = expected_indexes - represented_indexes
    if missing_matrix:
        errors.append(
            "comparison_matrix does not structurally represent paper indexes: "
            + ",".join(str(index) for index in sorted(missing_matrix))
        )

    return StructuralCheckResult(passed=not errors, errors=errors)


def check_synthesis_result(
    result: SynthesisAgentResult,
    *,
    selected_paper_ids: list[str],
) -> StructuralCheckResult:
    errors: list[str] = []
    selected = list(dict.fromkeys(selected_paper_ids))
    selected_set = set(selected)

    try:
        report = SynthesisReport.model_validate(result.report)
    except ValidationError as exc:
        return StructuralCheckResult(
            passed=False,
            errors=[f"synthesis report is invalid: {exc}"],
        )

    if result.agent_run.agent_name != "synthesis_agent":
        errors.append("agent_run.agent_name is not synthesis_agent.")
    if "policy_applied" not in result.agent_run.details:
        errors.append("agent_run.details is missing policy_applied.")

    citation_ids = [citation.paper_id for citation in report.citations]
    unknown = sorted(set(citation_ids) - selected_set)
    if unknown:
        errors.append("citations contain unknown paper ids: " + ",".join(unknown))

    missing = sorted(selected_set - set(citation_ids))
    if missing:
        errors.append("selected papers missing citations: " + ",".join(missing))

    return StructuralCheckResult(passed=not errors, errors=errors)
