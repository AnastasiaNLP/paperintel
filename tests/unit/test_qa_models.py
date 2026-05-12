import pytest
from pydantic import TypeAdapter, ValidationError

from models.qa import (
    AnswerDraft,
    CriticReview,
    EvidencePlan,
    Intent,
    IntentResolution,
    QAResult,
    RepairContext,
)
from models.retrieval import CitationRef
from models.session import Persona


def _citation() -> CitationRef:
    return CitationRef(
        paper_id="2310.06825",
        chunk_id="2310.06825:chunk:0",
        page_start=1,
        page_end=1,
        section_title="Method",
    )


def test_intent_literal_values_cover_stage_d_plan():
    adapter = TypeAdapter(Intent)

    for value in [
        "qa_factual",
        "qa_math",
        "qa_comparison",
        "qa_followup",
        "discover",
        "analyze_paper",
        "select_papers",
        "clarification_needed",
        "unclear",
    ]:
        assert adapter.validate_python(value) == value

    with pytest.raises(ValidationError):
        adapter.validate_python("simple_rag")


def test_persona_literal_values_match_session_contract():
    adapter = TypeAdapter(Persona)

    assert adapter.validate_python("engineer") == "engineer"
    assert adapter.validate_python("researcher") == "researcher"
    assert adapter.validate_python("techlead") == "techlead"

    with pytest.raises(ValidationError):
        adapter.validate_python("default")


def test_intent_resolution_default_id_unique():
    first = IntentResolution(intent="qa_factual")
    second = IntentResolution(intent="qa_factual")

    assert first.id != second.id
    assert first.confidence == 1.0
    assert first.referenced_paper_ids == []


def test_intent_resolution_ambiguous_requires_clarification():
    with pytest.raises(ValidationError):
        IntentResolution(intent="unclear", ambiguous=True)

    resolution = IntentResolution(
        intent="clarification_needed",
        ambiguous=True,
        clarification_question="Which paper should I use?",
    )

    assert resolution.ambiguous is True
    assert resolution.clarification_question == "Which paper should I use?"


def test_evidence_plan_tracks_chunk_types_and_sections():
    plan = EvidencePlan(
        intent="qa_math",
        paper_ids=["2310.06825"],
        search_query="loss function",
        chunk_types_priority=["equation", "text"],
        section_queries=[" Method ", "", "Results"],
    )

    assert plan.k == 8
    assert plan.chunk_types_priority == ["equation", "text"]
    assert plan.section_queries == ["Method", "Results"]


def test_evidence_plan_requires_paper_ids_and_positive_k():
    with pytest.raises(ValidationError):
        EvidencePlan(intent="qa_factual", paper_ids=[], search_query="latency")

    with pytest.raises(ValidationError):
        EvidencePlan(
            intent="qa_factual",
            paper_ids=["2310.06825"],
            search_query="latency",
            k=0,
        )


def test_answer_draft_defaults_and_flags():
    draft = AnswerDraft(
        question="What does the method optimize?",
        answer_text="It optimizes retrieval quality.",
        persona="researcher",
        citations=[_citation()],
    )

    assert draft.repair_iteration == 0
    assert draft.insufficient_evidence is False
    assert draft.limitations_noted is False
    assert draft.citations[0].chunk_id == "2310.06825:chunk:0"


def test_answer_draft_validates_confidence_and_repair_iteration():
    with pytest.raises(ValidationError):
        AnswerDraft(
            question="Question?",
            answer_text="Answer.",
            persona="engineer",
            confidence=1.2,
        )

    with pytest.raises(ValidationError):
        AnswerDraft(
            question="Question?",
            answer_text="Answer.",
            persona="engineer",
            repair_iteration=-1,
        )


def test_critic_review_repair_target_consistency():
    with pytest.raises(ValidationError):
        CriticReview(reviewed_answer_id="answer-1", needs_repair=True)

    review = CriticReview(
        reviewed_answer_id="answer-1",
        unsupported_claims=["Claim lacks evidence."],
        needs_repair=True,
        repair_target_agent="answer_agent",
        repair_instructions=["Remove the unsupported claim."],
    )

    assert review.needs_repair is True
    assert review.repair_target_agent == "answer_agent"


def test_repair_context_iteration_required_and_positive():
    with pytest.raises(ValidationError):
        RepairContext(
            original_run_id="run-1",
            target_agent="answer_agent",
            instructions=["Fix citations."],
            iteration=0,
            critic_review_id="review-1",
        )

    context = RepairContext(
        original_run_id="run-1",
        target_agent="answer_agent",
        instructions=[" Fix citations. ", ""],
        iteration=1,
        critic_review_id="review-1",
    )

    assert context.instructions == ["Fix citations."]
    assert context.iteration == 1


def test_qa_result_serializable_to_json_roundtrip():
    result = QAResult(
        session_id="session-1",
        question="What is the method?",
        answer="The method uses retrieval.",
        citations=[_citation()],
        persona="techlead",
        confidence=0.7,
        intent="qa_factual",
        agent_run_ids=["run-1", "run-2"],
    )

    reloaded = QAResult.model_validate_json(result.model_dump_json())

    assert reloaded == result
    assert reloaded.citations[0].section_title == "Method"


def test_qa_result_with_zero_citations_valid_for_insufficient_evidence():
    result = QAResult(
        session_id="session-1",
        question="What is the deployment cost?",
        answer="The indexed evidence is insufficient to answer that.",
        persona="engineer",
        confidence=0.2,
        intent="qa_factual",
        insufficient_evidence=True,
    )

    assert result.citations == []
    assert result.insufficient_evidence is True
