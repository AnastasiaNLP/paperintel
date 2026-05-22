from __future__ import annotations

from evaluation.golden_dataset import GoldenDatasetRecord
from evaluation.judge_models import JudgeResult, JudgeRunReport, JudgeTask
from evaluation.judge_rubrics import JudgeRubric
from models.artifacts import PaperWorkspace


REPORT_RUBRIC_IDS = [
    "recommended_action",
    "implementation_difficulty",
    "action_reasoning",
]


def build_dry_run_judge_report(
    records: list[GoldenDatasetRecord],
    workspaces: list[PaperWorkspace],
    rubrics: dict[str, JudgeRubric],
) -> JudgeRunReport:
    workspace_by_paper_id = {workspace.paper_id: workspace for workspace in workspaces}
    results: list[JudgeResult] = []

    for record in records:
        workspace = workspace_by_paper_id.get(record.paper_id)
        if workspace is None:
            continue
        for rubric_id in REPORT_RUBRIC_IDS:
            rubric = rubrics[rubric_id]
            task = JudgeTask(
                rubric_id=rubric.rubric_id,
                paper_id=record.paper_id,
                input_refs=_report_input_refs(record.paper_id),
                rubric_hash=rubric.sha256,
                mode="dry_run",
            )
            results.append(
                JudgeResult(
                    task=task,
                    status="not_scored",
                    rationale="Dry run only; no LLM judge was called.",
                )
            )

    return JudgeRunReport(
        mode="dry_run",
        total_tasks=len(results),
        scored_tasks=0,
        results=results,
    )


def _report_input_refs(paper_id: str) -> list[str]:
    return [
        f"paper_workspace:{paper_id}:finalized_report_json",
        f"paper_workspace:{paper_id}:method_extraction_json",
        f"paper_workspace:{paper_id}:benchmarks_json",
        f"paper_workspace:{paper_id}:readiness_json",
    ]

