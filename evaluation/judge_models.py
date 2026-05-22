from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JudgeMode = Literal["dry_run", "live"]
JudgeStatus = Literal["not_scored", "scored", "skipped", "error"]


class JudgeTask(BaseModel):
    rubric_id: str
    paper_id: str
    input_refs: list[str] = Field(default_factory=list)
    rubric_hash: str
    mode: JudgeMode = "dry_run"


class JudgeResult(BaseModel):
    task: JudgeTask
    status: JudgeStatus
    score: float | None = None
    rationale: str | None = None


class JudgeRunReport(BaseModel):
    mode: JudgeMode
    total_tasks: int
    scored_tasks: int
    results: list[JudgeResult] = Field(default_factory=list)

