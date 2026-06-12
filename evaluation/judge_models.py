from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


JudgeMode = Literal["dry_run", "live"]
JudgeStatus = Literal["not_scored", "scored", "skipped", "error"]
JudgeTaskFamily = Literal["report", "qa", "comparison", "synthesis"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JudgeTask(BaseModel):
    rubric_id: str
    paper_id: str
    sample_id: str | None = None
    task_family: JudgeTaskFamily = "report"
    input_refs: list[str] = Field(default_factory=list)
    rubric_hash: str
    rubric_version: str | None = None
    mode: JudgeMode = "dry_run"
    judge_model: str | None = None
    dataset_version: str | None = None
    pipeline_version: str | None = None


class JudgeResult(BaseModel):
    task: JudgeTask
    status: JudgeStatus
    score: float | None = None
    rationale: str | None = None
    error_code: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class JudgeRunReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    mode: JudgeMode
    judge_model: str | None = None
    dataset_version: str | None = None
    pipeline_version: str | None = None
    rubric_versions: dict[str, str] = Field(default_factory=dict)
    total_tasks: int
    scored_tasks: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    average_scores_by_rubric: dict[str, float] = Field(default_factory=dict)
    average_score: float | None = None
    results: list[JudgeResult] = Field(default_factory=list)


class JudgeBaselineDelta(BaseModel):
    task_family: JudgeTaskFamily = "report"
    sample_id: str
    rubric_id: str
    current_score: float
    baseline_score: float
    delta: float


class JudgeBaselineComparison(BaseModel):
    current_count: int
    baseline_count: int
    matched_scored_tasks: int
    improved: list[JudgeBaselineDelta] = Field(default_factory=list)
    regressed: list[JudgeBaselineDelta] = Field(default_factory=list)
    unchanged: list[JudgeBaselineDelta] = Field(default_factory=list)
    missing_in_current: list[str] = Field(default_factory=list)
    missing_in_baseline: list[str] = Field(default_factory=list)
