from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from evaluation.golden_dataset import GoldenDatasetRecord
from evaluation.judge_models import JudgeResult, JudgeTask
from evaluation.judge_rubrics import JudgeRubric
from models.artifacts import PaperWorkspace


class JudgePayload(BaseModel):
    paper_id: str
    title: str | None = None
    rubric_id: str
    rubric_text: str
    finalized_report_json: dict[str, Any] | None = None
    method_extraction_json: dict[str, Any] | None = None
    benchmarks_json: list[dict[str, Any]] = Field(default_factory=list)
    readiness_json: dict[str, Any] | None = None
    golden_report_coverage: dict[str, Any] | None = None


class JudgeProvider(Protocol):
    def score(
        self,
        *,
        task: JudgeTask,
        rubric: JudgeRubric,
        payload: JudgePayload,
    ) -> JudgeResult:
        ...


class DryRunJudgeProvider:
    def score(
        self,
        *,
        task: JudgeTask,
        rubric: JudgeRubric,
        payload: JudgePayload,
    ) -> JudgeResult:
        return JudgeResult(
            task=task,
            status="not_scored",
            rationale="Dry run only; no LLM judge was called.",
        )


class ConfiguredLLMJudgeProvider:
    def __init__(
        self,
        *,
        requested_model: str | None = None,
        max_tokens: int = 700,
    ) -> None:
        self.requested_model = requested_model
        self.max_tokens = max_tokens

    def score(
        self,
        *,
        task: JudgeTask,
        rubric: JudgeRubric,
        payload: JudgePayload,
    ) -> JudgeResult:
        from agents.llm_provider import call_text_llm

        raw, error = call_text_llm(
            requested_model=self.requested_model,
            system_prompt=_judge_system_prompt(),
            user_content=_judge_user_content(payload),
            max_tokens=self.max_tokens,
            context_label=f"judge_{task.rubric_id}",
        )
        if error:
            return JudgeResult(
                task=task,
                status="error",
                rationale=error,
            )

        return _parse_judge_response(task, raw or "")


def build_report_judge_payload(
    *,
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace,
    rubric: JudgeRubric,
) -> JudgePayload:
    return JudgePayload(
        paper_id=record.paper_id,
        title=record.title,
        rubric_id=rubric.rubric_id,
        rubric_text=rubric.text,
        finalized_report_json=workspace.finalized_report_json,
        method_extraction_json=workspace.method_extraction_json,
        benchmarks_json=workspace.benchmarks_json,
        readiness_json=workspace.readiness_json,
        golden_report_coverage=record.expected_report_coverage.model_dump(),
    )


def _judge_system_prompt() -> str:
    return (
        "You are an evaluation judge for PaperIntel. Score only from the supplied "
        "rubric and payload. Do not use outside knowledge. Return strict JSON "
        "with keys: score, rationale. The score must be a number from 0.0 to 1.0."
    )


def _judge_user_content(payload: JudgePayload) -> str:
    return json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True)


def _parse_judge_response(task: JudgeTask, raw: str) -> JudgeResult:
    try:
        payload = json.loads(_extract_json_object(raw))
        score = float(payload["score"])
        rationale = str(payload.get("rationale") or "").strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return JudgeResult(
            task=task,
            status="error",
            rationale=f"Judge returned invalid JSON: {exc}; raw={raw[:500]}",
        )

    if not 0.0 <= score <= 1.0:
        return JudgeResult(
            task=task,
            status="error",
            rationale=f"Judge score out of range: {score}",
        )

    return JudgeResult(
        task=task,
        status="scored",
        score=score,
        rationale=rationale,
    )


def _extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    return stripped[start : end + 1]

