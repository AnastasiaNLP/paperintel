import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field


JobKind: TypeAlias = Literal[
    "analyze_paper",
    "analyze_selected",
    "analyze_pdf_blob",
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


def pdf_blob_idempotency_key(
    *,
    session_id: str,
    blob_id: str,
    paper_id: str | None,
    pipeline_version: str,
) -> str:
    payload = json.dumps(
        [session_id, blob_id, paper_id, pipeline_version],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"analyze_pdf_blob:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


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
    idempotency_key: str | None = None
    pipeline_version: str = "v1"
    next_attempt_at: datetime | None = None
    retry_policy_json: dict[str, Any] = Field(default_factory=dict)
    locked_by: str | None = None
    locked_at: datetime | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
