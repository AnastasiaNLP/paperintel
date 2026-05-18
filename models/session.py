from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field

from models.agent_runs import AgentRun
from models.errors import StructuredError
from models.retrieval import CitationRef


Persona: TypeAlias = Literal["engineer", "researcher", "techlead"]
SessionPhase: TypeAlias = Literal[
    "idle",
    "discovery",
    "selection",
    "analysis",
    "comparison",
    "qa",
    "failed",
]
TurnRole: TypeAlias = Literal["user", "assistant", "system"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    persona: Persona = "engineer"
    original_query: str | None = None
    phase: SessionPhase = "idle"
    selected_candidate_ids: list[str] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
    latest_comparison_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: TurnRole
    content: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    error: StructuredError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class GraphInvocationResult(BaseModel):
    response_text: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    needs_analysis: bool = False
    needs_discovery: bool = False
    discovery_topic: str | None = None
    discovery_candidate_count: int | None = None
    selected_candidate_ids: list[str] = Field(default_factory=list)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    next_phase: SessionPhase | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class HandlerResult(BaseModel):
    session_id: str
    response_text: str
    phase: SessionPhase
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    needs_analysis: bool = False
    needs_discovery: bool = False
    discovery_topic: str | None = None
    discovery_candidate_count: int | None = None
    selected_candidate_ids: list[str] = Field(default_factory=list)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    user_turn_id: str
    assistant_turn_id: str
    error: StructuredError | None = None
