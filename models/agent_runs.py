from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field


AgentRunStatus: TypeAlias = Literal["running", "completed", "failed", "fallback_used"]
TerminationReason: TypeAlias = Literal[
    "success",
    "max_iter",
    "timeout",
    "budget",
    "fallback",
    "error",
    "skipped",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    job_id: str | None = None
    agent_name: str
    input_refs: list[str] = Field(default_factory=list)
    output_ref: str | None = None
    confidence: float | None = None
    model: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = 0
    llm_call_count: int = 0
    termination_reason: TerminationReason | None = None
    status: AgentRunStatus = "running"
    tokens_used: int | None = None
    cost_usd: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None

    def complete(
        self,
        *,
        output_ref: str | None = None,
        confidence: float | None = None,
        termination_reason: TerminationReason = "success",
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> "AgentRun":
        self.output_ref = output_ref
        self.confidence = confidence
        self.termination_reason = termination_reason
        self.tokens_used = tokens_used
        self.cost_usd = cost_usd
        if details:
            self.details.update(details)
        self.status = "completed"
        self.finished_at = utc_now()
        return self

    def fail(
        self,
        *,
        termination_reason: TerminationReason = "error",
        output_ref: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "AgentRun":
        self.output_ref = output_ref
        self.termination_reason = termination_reason
        if details:
            self.details.update(details)
        self.status = "failed"
        self.finished_at = utc_now()
        return self

    def fallback(
        self,
        *,
        output_ref: str | None = None,
        termination_reason: TerminationReason = "fallback",
        details: dict[str, Any] | None = None,
    ) -> "AgentRun":
        self.output_ref = output_ref
        self.termination_reason = termination_reason
        if details:
            self.details.update(details)
        self.status = "fallback_used"
        self.finished_at = utc_now()
        return self
