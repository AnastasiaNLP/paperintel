from operator import add as add_lists
from typing import Annotated, TypedDict

from models.agent_runs import AgentRun
from models.errors import StructuredError
from models.qa import (
    AnswerDraft,
    CriticReview,
    EvidencePlan,
    Intent,
    IntentResolution,
    RepairContext,
)
from models.retrieval import EvidenceBundle
from models.session import Persona


class ConversationState(TypedDict, total=False):
    session_id: str
    user_message: str
    persona: Persona

    intent_resolution: IntentResolution
    intent: Intent
    referenced_paper_ids: list[str]
    needs_clarification: bool
    clarification_question: str | None

    evidence_plan: EvidencePlan
    evidence_bundle: EvidenceBundle
    evidence_bundle_ref: str | None

    answer_draft: AnswerDraft

    critic_review: CriticReview
    repair_context: RepairContext | None

    agent_runs: Annotated[list[AgentRun], add_lists]
    errors: Annotated[list[StructuredError], add_lists]
