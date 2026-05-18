from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from models.api import HealthStatus
from models.session import HandlerResult, Persona, Session, Turn


class CreateSessionRequest(BaseModel):
    persona: Persona = "engineer"
    original_query: str | None = None


class AnalyzeRequest(BaseModel):
    paper_url: HttpUrl


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DiscoverRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)


class SelectPapersRequest(BaseModel):
    selection: str = Field(min_length=1, max_length=500)


class SynthesizeRequest(BaseModel):
    prompt: str | None = Field(default=None, max_length=2000)


class SessionResponse(BaseModel):
    id: str
    persona: Persona
    phase: str
    active_paper_ids: list[str]
    original_query: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_session(cls, session: Session) -> "SessionResponse":
        return cls(
            id=session.id,
            persona=session.persona,
            phase=session.phase,
            active_paper_ids=session.active_paper_ids,
            original_query=session.original_query,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class MessageResponse(BaseModel):
    session_id: str
    response_text: str
    phase: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    comparison_markdown: str | None = None
    needs_analysis: bool = False
    needs_discovery: bool = False
    discovery_topic: str | None = None
    discovery_candidate_count: int | None = None
    selected_candidate_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_handler_result(cls, result: HandlerResult) -> "MessageResponse":
        return cls(
            session_id=result.session_id,
            response_text=result.response_text,
            phase=result.phase,
            intent=result.intent,
            referenced_paper_ids=result.referenced_paper_ids,
            citations=[citation.model_dump(mode="json") for citation in result.citations],
            artifact_refs=result.artifact_refs,
            comparison_markdown=result.comparison_markdown,
            needs_analysis=result.needs_analysis,
            needs_discovery=result.needs_discovery,
            discovery_topic=result.discovery_topic,
            discovery_candidate_count=result.discovery_candidate_count,
            selected_candidate_ids=result.selected_candidate_ids,
        )


class TurnResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def from_turn(cls, turn: Turn) -> "TurnResponse":
        return cls(
            id=turn.id,
            session_id=turn.session_id,
            role=turn.role,
            content=turn.content,
            intent=turn.intent,
            referenced_paper_ids=turn.referenced_paper_ids,
            artifact_refs=turn.artifact_refs,
            created_at=turn.created_at,
        )


class TurnsResponse(BaseModel):
    turns: list[TurnResponse]


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_health_status(cls, status: HealthStatus) -> "HealthResponse":
        return cls(
            status="healthy" if status.healthy else "degraded",
            checks=status.checks,
        )


class ErrorResponse(BaseModel):
    error: str
    detail: str
