from __future__ import annotations

from typing import Any

from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.synthesis import SynthesisAgentResult


def build_comparison_judge_payload(
    *,
    artifact: ComparisonArtifact,
    workspaces: list[PaperWorkspace],
) -> dict[str, Any]:
    selected_ids = list(dict.fromkeys(artifact.paper_ids))
    workspace_by_id = {workspace.paper_id: workspace for workspace in workspaces}
    _require_selected_workspaces(selected_ids, workspace_by_id)
    return {
        "task_type": "comparison_analyst",
        "selected_paper_ids": selected_ids,
        "workspaces": [
            _workspace_payload(workspace_by_id[paper_id])
            for paper_id in selected_ids
        ],
        "comparison_artifact": {
            "id": artifact.id,
            "paper_ids": artifact.paper_ids,
            "comparison_report_json": artifact.comparison_report_json,
            "comparison_markdown": artifact.comparison_markdown,
        },
    }


def build_synthesis_judge_payload(
    *,
    result: SynthesisAgentResult,
    workspaces: list[PaperWorkspace],
    comparison: ComparisonArtifact | None = None,
) -> dict[str, Any]:
    selected_ids = _paper_ids_from_input_refs(result.agent_run.input_refs)
    workspace_by_id = {workspace.paper_id: workspace for workspace in workspaces}
    _require_selected_workspaces(selected_ids, workspace_by_id)
    payload: dict[str, Any] = {
        "task_type": "synthesis_agent",
        "persona": result.report.persona,
        "selected_paper_ids": selected_ids,
        "workspaces": [
            _workspace_payload(workspace_by_id[paper_id])
            for paper_id in selected_ids
        ],
        "synthesis_report": result.report.model_dump(mode="json"),
        "response_text": result.response_text,
        "agent_run": {
            "agent_name": result.agent_run.agent_name,
            "input_refs": result.agent_run.input_refs,
            "details": result.agent_run.details,
            "status": result.agent_run.status,
        },
    }
    if comparison is not None and set(comparison.paper_ids).intersection(selected_ids):
        payload["comparison_context"] = {
            "id": comparison.id,
            "paper_ids": comparison.paper_ids,
            "comparison_report_json": comparison.comparison_report_json,
            "comparison_markdown": comparison.comparison_markdown,
        }
    return payload


def _require_selected_workspaces(
    selected_ids: list[str],
    workspace_by_id: dict[str, PaperWorkspace],
) -> None:
    missing = [
        paper_id
        for paper_id in selected_ids
        if paper_id not in workspace_by_id
    ]
    if missing:
        raise ValueError(
            "Missing workspaces for selected paper ids: " + ",".join(missing)
        )


def _paper_ids_from_input_refs(input_refs: list[str]) -> list[str]:
    paper_ids = [
        ref.removeprefix("paper_workspace:")
        for ref in input_refs
        if ref.startswith("paper_workspace:")
    ]
    return list(dict.fromkeys(paper_ids))


def _workspace_payload(workspace: PaperWorkspace) -> dict[str, Any]:
    return {
        "paper_id": workspace.paper_id,
        "title": workspace.title,
        "source_url": workspace.source_url,
        "pipeline_stage": workspace.pipeline_stage,
        "finalized_report_json": workspace.finalized_report_json,
        "method_extraction_json": workspace.method_extraction_json,
        "benchmarks_json": workspace.benchmarks_json,
        "readiness_json": workspace.readiness_json,
        "full_markdown_report": workspace.full_markdown_report,
    }
