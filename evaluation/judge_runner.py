from __future__ import annotations

from evaluation.judge_automation import finalize_judge_report
from evaluation.golden_dataset import GoldenDatasetRecord
from evaluation.judge_models import JudgeResult, JudgeRunReport, JudgeTask
from evaluation.judge_provider import (
    DryRunJudgeProvider,
    JudgeProvider,
    build_report_judge_payload,
)
from evaluation.judge_rubrics import JudgeRubric
from evaluation.ca_judge_payloads import (
    build_comparison_judge_payload,
    build_synthesis_judge_payload,
)
from evaluation.judge_provider import JudgePayload
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.synthesis import SynthesisAgentResult, SynthesisReport


REPORT_RUBRIC_IDS = [
    "recommended_action",
    "implementation_difficulty",
    "action_reasoning",
]
QA_RUBRIC_IDS = ["qa_faithfulness"]
COMPARISON_RUBRIC_IDS = [
    "comparison_balance",
    "comparison_grounding",
    "comparison_recommendation_justification",
]
SYNTHESIS_RUBRIC_IDS = [
    "synthesis_persona_fit",
    "synthesis_justification",
    "synthesis_citation_faithfulness",
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
    judge_model: str | None = None,
    dataset_version: str | None = None,
    pipeline_version: str | None = None,
    qa_samples: list[dict] | None = None,
    comparisons: list[dict] | None = None,
    syntheses: list[dict] | None = None,
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
                sample_id=f"report:{record.paper_id}",
                task_family="report",
                input_refs=_report_input_refs(record.paper_id),
                rubric_hash=rubric.sha256,
                rubric_version=_rubric_version(rubric),
                mode=mode,
                judge_model=judge_model,
                dataset_version=dataset_version,
                pipeline_version=pipeline_version,
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
            results.append(_score(provider, task, rubric, payload))

    if qa_samples is not None:
        valid_qa_keys = {
            (record.paper_id, qa_case.id)
            for record in records
            for qa_case in record.qa_cases
        }
        qa_samples_by_key: dict[tuple[str, str], dict] = {}
        for item in qa_samples:
            key = (
                str(item.get("paper_id", "")).strip(),
                str(item.get("qa_case_id", "")).strip(),
            )
            if not all(key):
                raise ValueError("QA samples require non-empty paper_id and qa_case_id")
            if key not in valid_qa_keys:
                raise ValueError(
                    f"QA sample does not match a golden QA case: {key[0]}:{key[1]}"
                )
            if key in qa_samples_by_key:
                raise ValueError(f"Duplicate QA sample: {key[0]}:{key[1]}")
            if not isinstance(item.get("answer_text"), str):
                raise ValueError(f"QA sample {key[0]}:{key[1]} requires answer_text")
            if not isinstance(item.get("citations", []), list):
                raise ValueError(f"QA sample {key[0]}:{key[1]} citations must be a list")
            if not isinstance(item.get("evidence_chunks", []), list):
                raise ValueError(
                    f"QA sample {key[0]}:{key[1]} evidence_chunks must be a list"
                )
            qa_samples_by_key[key] = item
        for record in records:
            for qa_case in record.qa_cases:
                sample = qa_samples_by_key.get((record.paper_id, qa_case.id))
                for rubric_id in QA_RUBRIC_IDS:
                    rubric = rubrics[rubric_id]
                    task = _task(
                        rubric=rubric,
                        paper_id=record.paper_id,
                        sample_id=f"qa:{record.paper_id}:{qa_case.id}",
                        task_family="qa",
                        input_refs=[
                            f"golden_qa:{record.paper_id}:{qa_case.id}",
                            f"qa_sample:{record.paper_id}:{qa_case.id}",
                        ],
                        mode=mode,
                        judge_model=judge_model,
                        dataset_version=dataset_version,
                        pipeline_version=pipeline_version,
                    )
                    if sample is None:
                        results.append(_skipped(task, "No QA result sample was supplied."))
                        continue
                    payload = JudgePayload(
                        paper_id=record.paper_id,
                        title=record.title,
                        rubric_id=rubric.rubric_id,
                        rubric_text=rubric.text,
                        additional_context={
                            "question": qa_case.question,
                            "answer_text": sample.get("answer_text", ""),
                            "citations": sample.get("citations", []),
                            "evidence_chunks": sample.get("evidence_chunks", []),
                            "golden_qa_case": qa_case.model_dump(mode="json"),
                        },
                    )
                    results.append(_score(provider, task, rubric, payload))

    for item in comparisons or []:
        artifact = ComparisonArtifact.model_validate(item.get("artifact", item))
        if len(set(artifact.paper_ids)) < 2:
            raise ValueError(
                f"Comparison artifact {artifact.id} must include at least two papers"
            )
        input_workspaces = item.get("workspaces", [])
        selected_workspaces = [PaperWorkspace.model_validate(row) for row in input_workspaces]
        payload_data = build_comparison_judge_payload(
            artifact=artifact,
            workspaces=selected_workspaces,
        )
        for rubric_id in COMPARISON_RUBRIC_IDS:
            rubric = rubrics[rubric_id]
            task = _task(
                rubric=rubric,
                paper_id=artifact.paper_ids[0],
                sample_id=f"comparison:{artifact.id}",
                task_family="comparison",
                input_refs=[f"comparison_artifact:{artifact.id}"]
                + [f"paper_workspace:{pid}" for pid in artifact.paper_ids],
                mode=mode,
                judge_model=judge_model,
                dataset_version=dataset_version,
                pipeline_version=pipeline_version,
            )
            payload = JudgePayload(
                paper_id=artifact.paper_ids[0],
                title=None,
                rubric_id=rubric.rubric_id,
                rubric_text=rubric.text,
                additional_context=payload_data,
            )
            results.append(_score(provider, task, rubric, payload))

    for item in syntheses or []:
        report = SynthesisReport.model_validate(item["report"])
        run = AgentRun.model_validate(item["agent_run"])
        synthesis = SynthesisAgentResult(
            report=report,
            response_text=str(item.get("response_text", "")),
            agent_run=run,
        )
        input_workspaces = [PaperWorkspace.model_validate(row) for row in item.get("workspaces", [])]
        comparison_data = item.get("comparison")
        comparison = (
            ComparisonArtifact.model_validate(comparison_data)
            if comparison_data
            else None
        )
        payload_data = build_synthesis_judge_payload(
            result=synthesis,
            workspaces=input_workspaces,
            comparison=comparison,
        )
        paper_ids = [
            ref.removeprefix("paper_workspace:")
            for ref in run.input_refs
            if ref.startswith("paper_workspace:")
        ]
        if not paper_ids:
            raise ValueError("Synthesis judge input must include paper_workspace refs")
        sample_id = f"synthesis:{run.id}"
        for rubric_id in SYNTHESIS_RUBRIC_IDS:
            rubric = rubrics[rubric_id]
            task = _task(
                rubric=rubric,
                paper_id=paper_ids[0],
                sample_id=sample_id,
                task_family="synthesis",
                input_refs=[f"agent_run:{run.id}"]
                + [f"paper_workspace:{pid}" for pid in paper_ids],
                mode=mode,
                judge_model=judge_model,
                dataset_version=dataset_version,
                pipeline_version=pipeline_version,
            )
            payload = JudgePayload(
                paper_id=paper_ids[0],
                rubric_id=rubric.rubric_id,
                rubric_text=rubric.text,
                additional_context=payload_data,
            )
            results.append(_score(provider, task, rubric, payload))

    return finalize_judge_report(
        JudgeRunReport(
            mode=mode,
            judge_model=judge_model,
            dataset_version=dataset_version,
            pipeline_version=pipeline_version,
            rubric_versions={
                rubric_id: _rubric_version(rubric)
                for rubric_id, rubric in sorted(rubrics.items())
            },
            total_tasks=len(results),
            scored_tasks=sum(1 for result in results if result.status == "scored"),
            results=results,
        )
    )


def _task(
    *, rubric, paper_id, sample_id, task_family, input_refs, mode,
    judge_model, dataset_version, pipeline_version,
) -> JudgeTask:
    return JudgeTask(
        rubric_id=rubric.rubric_id,
        paper_id=paper_id,
        sample_id=sample_id,
        task_family=task_family,
        input_refs=input_refs,
        rubric_hash=rubric.sha256,
        rubric_version=_rubric_version(rubric),
        mode=mode,
        judge_model=judge_model,
        dataset_version=dataset_version,
        pipeline_version=pipeline_version,
    )


def _skipped(task: JudgeTask, reason: str) -> JudgeResult:
    return JudgeResult(task=task, status="skipped", rationale=reason)


def _score(provider, task, rubric, payload) -> JudgeResult:
    try:
        return provider.score(task=task, rubric=rubric, payload=payload)
    except Exception as exc:
        return JudgeResult(
            task=task,
            status="error",
            rationale=f"Judge provider failed: {type(exc).__name__}",
            error_code="judge_provider_failed",
        )


def _report_input_refs(paper_id: str) -> list[str]:
    return [
        f"paper_workspace:{paper_id}:finalized_report_json",
        f"paper_workspace:{paper_id}:method_extraction_json",
        f"paper_workspace:{paper_id}:benchmarks_json",
        f"paper_workspace:{paper_id}:readiness_json",
    ]


def _rubric_version(rubric: JudgeRubric) -> str:
    return f"sha256:{rubric.sha256[:12]}"
