from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field


JobKind: TypeAlias = Literal[
    "analyze_paper",
    "analyze_selected",
    "discover",
    "compare",
    "synthesize",
    "judge_eval",
]
JobStatus: TypeAlias = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    kind: JobKind
    status: JobStatus = "queued"
    input_json: dict[str, Any] = Field(default_factory=dict)
    result_json: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    locked_by: str | None = None
    locked_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
