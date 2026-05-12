from datetime import datetime, timezone
from typing import Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from models.retrieval import ChunkType, CitationRef
from models.session import Persona


Intent: TypeAlias = Literal[
    "qa_factual",
    "qa_math",
    "qa_comparison",
    "qa_followup",
    "discover",
    "analyze_paper",
    "select_papers",
    "clarification_needed",
    "unclear",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntentResolution(BaseModel):
    """Output contract for the Intent Router agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    intent: Intent
    referenced_paper_ids: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    clarification_question: str | None = None
    confidence: float = 1.0
    reasoning: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def ambiguous_resolution_requires_clarification(self) -> "IntentResolution":
        if self.ambiguous and not self.clarification_question:
            raise ValueError("ambiguous intent requires clarification_question")
        return self


class EvidencePlan(BaseModel):
    """Output contract for the Evidence Retrieval Planner agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    intent: Intent
    paper_ids: list[str]
    search_query: str
    chunk_types_priority: list[ChunkType] = Field(default_factory=list)
    section_queries: list[str] = Field(default_factory=list)
    k: int = 8
    requires_replanning: bool = False
    replanning_reason: str | None = None
    fallback_used: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("paper_ids")
    @classmethod
    def paper_ids_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("paper_ids must not be empty")
        return value

    @field_validator("search_query")
    @classmethod
    def search_query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("search_query must not be blank")
        return value

    @field_validator("section_queries")
    @classmethod
    def normalize_section_queries(cls, value: list[str]) -> list[str]:
        return [section.strip() for section in value if section.strip()]

    @field_validator("k")
    @classmethod
    def k_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("k must be positive")
        return value


class AnswerDraft(BaseModel):
    """Output contract for the Answer agent before critic approval."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer_text: str
    citations: list[CitationRef] = Field(default_factory=list)
    persona: Persona
    confidence: float = 1.0
    limitations_noted: bool = False
    insufficient_evidence: bool = False
    repair_iteration: int = 0
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("question", "answer_text")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("repair_iteration")
    @classmethod
    def repair_iteration_must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repair_iteration must not be negative")
        return value


class CriticReview(BaseModel):
    """Output contract for the Citation Critic agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    reviewed_answer_id: str
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence_adjustments: dict[str, float] = Field(default_factory=dict)
    needs_repair: bool = False
    repair_target_agent: str | None = None
    repair_instructions: list[str] = Field(default_factory=list)
    critic_confidence: float = 1.0
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("reviewed_answer_id")
    @classmethod
    def reviewed_answer_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reviewed_answer_id must not be blank")
        return value

    @field_validator("critic_confidence")
    @classmethod
    def critic_confidence_must_be_probability(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("critic_confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def repair_review_requires_target_and_instructions(self) -> "CriticReview":
        if self.needs_repair:
            if not self.repair_target_agent:
                raise ValueError("repair_target_agent is required when needs_repair")
            if not self.repair_instructions:
                raise ValueError("repair_instructions are required when needs_repair")
        return self


class RepairContext(BaseModel):
    """Bounded repair instructions passed from a critic to a target agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    original_run_id: str
    target_agent: str
    instructions: list[str]
    iteration: int
    critic_review_id: str

    @field_validator(
        "original_run_id",
        "target_agent",
        "critic_review_id",
    )
    @classmethod
    def ids_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("id fields must not be blank")
        return value

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_empty(cls, value: list[str]) -> list[str]:
        normalized = [instruction.strip() for instruction in value if instruction.strip()]
        if not normalized:
            raise ValueError("instructions must not be empty")
        return normalized

    @field_validator("iteration")
    @classmethod
    def iteration_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("iteration must be positive")
        return value


class QAResult(BaseModel):
    """Final output contract returned by the conversation QA flow."""

    session_id: str
    question: str
    answer: str
    citations: list[CitationRef] = Field(default_factory=list)
    persona: Persona
    confidence: float
    intent: Intent
    insufficient_evidence: bool = False
    repair_iterations_used: int = 0
    agent_run_ids: list[str] = Field(default_factory=list)

    @field_validator("session_id", "question", "answer")
    @classmethod
    def required_text_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text fields must not be blank")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("repair_iterations_used")
    @classmethod
    def repair_iterations_used_must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repair_iterations_used must not be negative")
        return value
