from __future__ import annotations

from evaluation.golden_dataset import GoldenDatasetRecord
from evaluation.judge_models import JudgeResult, JudgeRunReport, JudgeTask
from evaluation.judge_provider import (
    DryRunJudgeProvider,
    JudgeProvider,
    build_report_judge_payload,
)
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
    return build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=DryRunJudgeProvider(),
        mode="dry_run",
    )


def build_judge_report(
    *,
    records: list[GoldenDatasetRecord],
    workspaces: list[PaperWorkspace],
    rubrics: dict[str, JudgeRubric],
    provider: JudgeProvider,
    mode: str,
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
                mode=mode,
            )
            if workspace.finalized_report_json is None:
                results.append(
                    JudgeResult(
                        task=task,
                        status="skipped",
                        rationale=(
                            "No finalized_report_json is available for report "
                            "rubric evaluation."
                        ),
                    )
                )
                continue
            payload = build_report_judge_payload(
                record=record,
                workspace=workspace,
                rubric=rubric,
            )
            results.append(provider.score(task=task, rubric=rubric, payload=payload))

    return JudgeRunReport(
        mode=mode,
        total_tasks=len(results),
        scored_tasks=sum(1 for result in results if result.status == "scored"),
        results=results,
    )


def _report_input_refs(paper_id: str) -> list[str]:
    return [
        f"paper_workspace:{paper_id}:finalized_report_json",
        f"paper_workspace:{paper_id}:method_extraction_json",
        f"paper_workspace:{paper_id}:benchmarks_json",
        f"paper_workspace:{paper_id}:readiness_json",
    ]
