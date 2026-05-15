import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.citation_critic import citation_critic_agent
from models.agent_policies import AgentRuntimePolicy
from models.agent_runs import AgentRun
from models.qa import AnswerDraft
from models.retrieval import (
    ChunkLocation,
    ChunkSearchResult,
    ChunkSource,
    CitationRef,
    EvidenceBundle,
    PaperChunk,
)
from services.repair import MAX_REPAIR_ITERATIONS


def _chunk() -> PaperChunk:
    return PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="The method improves retrieval quality on the benchmark.",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id="session-1",
            arxiv_id="2310.06825",
            title="Example Paper",
        ),
        location=ChunkLocation(page_start=2, page_end=2, section_title="Method"),
    )


def _evidence() -> EvidenceBundle:
    chunk = _chunk()
    return EvidenceBundle(
        query="retrieval quality",
        results=[ChunkSearchResult(chunk=chunk, score=0.91, rank=1)],
    )


def _answer(**overrides) -> AnswerDraft:
    data = {
        "id": "answer-1",
        "question": "What does the method improve?",
        "answer_text": "The method improves retrieval quality.",
        "persona": "engineer",
        "confidence": 0.8,
        "citations": [
            CitationRef(
                paper_id="2310.06825",
                chunk_id="2310.06825:chunk:0",
                page_start=2,
                page_end=2,
                section_title="Method",
            )
        ],
    }
    data.update(overrides)
    return AnswerDraft(**data)


def _state(**overrides) -> dict:
    answer_run = AgentRun(agent_name="answer_agent", session_id="session-1")
    state = {
        "session_id": "session-1",
        "answer_draft": _answer(),
        "evidence_bundle": _evidence(),
        "agent_runs": [answer_run],
    }
    state.update(overrides)
    return state


def _config(persistence=None) -> dict:
    return {
        "configurable": {
            "session_id": "session-1",
            "job_id": "job-1",
            "agent_run_persistence": persistence or InMemoryAgentRunPersistence(),
        }
    }


def _review_payload(**overrides) -> str:
    payload = {
        "unsupported_claims": [],
        "missing_evidence": [],
        "contradictions": [],
        "confidence_adjustments": {},
        "needs_repair": False,
        "repair_instructions": [],
        "critic_confidence": 0.9,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


@patch("agents.citation_critic._call_llm")
def test_citation_critic_accepts_grounded_answer(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_review_payload(), None)

    result = citation_critic_agent(_state(), config=_config(persistence))

    run = _run(result)
    review = result["critic_review"]
    assert review.reviewed_answer_id == "answer-1"
    assert review.needs_repair is False
    assert result["repair_context"] is None
    assert run.agent_name == "citation_critic"
    assert run.model == "claude-sonnet-4-6"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:critic_review"
    assert run.llm_call_count == 1
    assert run.details["needs_repair"] is False
    assert run.details["policy_applied"]["fallback_strategy"] == (
        "downgrade_after_repair_exhaustion"
    )
    assert persistence.list_runs() == [run]


def test_citation_critic_skips_insufficient_evidence_answer():
    result = citation_critic_agent(
        _state(
            answer_draft=_answer(
                citations=[],
                insufficient_evidence=True,
                confidence=0.0,
            )
        ),
        config=_config(),
    )

    run = _run(result)
    review = result["critic_review"]
    assert review.needs_repair is False
    assert result["repair_context"] is None
    assert run.status == "completed"
    assert run.termination_reason == "skipped"
    assert run.llm_call_count == 0
    assert run.details["fallback_reason"] == "insufficient_evidence_no_review_needed"


@patch("agents.citation_critic._call_llm")
def test_citation_critic_triggers_repair_on_unsupported_claims(mock_call_llm):
    mock_call_llm.return_value = (
        _review_payload(
            unsupported_claims=["The answer claims lower latency without evidence."],
            needs_repair=True,
            repair_instructions=["Remove the lower latency claim."],
        ),
        None,
    )

    result = citation_critic_agent(_state(), config=_config())

    run = _run(result)
    review = result["critic_review"]
    repair_context = result["repair_context"]
    assert review.needs_repair is True
    assert result["answer_draft"] is None
    assert repair_context.target_agent == "answer_agent"
    assert repair_context.instructions == ["Remove the lower latency claim."]
    assert repair_context.iteration == 1
    assert run.status == "completed"
    assert run.details["needs_repair"] is True


@patch("agents.citation_critic._call_llm")
def test_citation_critic_infers_repair_when_claim_lists_non_empty(mock_call_llm):
    mock_call_llm.return_value = (
        _review_payload(
            missing_evidence=["No chunk supports deployment cost."],
            needs_repair=False,
        ),
        None,
    )

    result = citation_critic_agent(_state(), config=_config())

    review = result["critic_review"]
    assert review.needs_repair is True
    assert review.repair_instructions
    assert result["repair_context"].target_agent == "answer_agent"


@patch("agents.citation_critic._call_llm")
def test_citation_critic_ignores_repair_flag_without_issues_or_instructions(
    mock_call_llm,
):
    mock_call_llm.return_value = (
        _review_payload(
            unsupported_claims=[],
            missing_evidence=[],
            contradictions=[],
            needs_repair=True,
            repair_instructions=[],
        ),
        None,
    )

    result = citation_critic_agent(_state(), config=_config())

    run = _run(result)
    review = result["critic_review"]
    assert review.needs_repair is False
    assert review.repair_instructions == []
    assert result["repair_context"] is None
    assert "answer_draft" not in result
    assert run.status == "completed"
    assert run.details["needs_repair"] is False
    assert run.details["unsupported_claims"] == 0
    assert run.details["missing_evidence"] == 0
    assert run.details["contradictions"] == 0


@patch("agents.citation_critic._call_llm")
def test_citation_critic_downgrades_after_max_repair_iterations(mock_call_llm):
    mock_call_llm.return_value = (
        _review_payload(
            unsupported_claims=["Unsupported claim remains."],
            needs_repair=True,
            repair_instructions=["Remove unsupported claim."],
        ),
        None,
    )

    result = citation_critic_agent(
        _state(answer_draft=_answer(repair_iteration=MAX_REPAIR_ITERATIONS)),
        config=_config(),
    )

    run = _run(result)
    downgraded = result["answer_draft"]
    assert result["repair_context"] is None
    assert downgraded.confidence == 0.3
    assert downgraded.limitations_noted is True
    assert "Limited evidence note" in downgraded.answer_text
    assert run.status == "fallback_used"
    assert run.termination_reason == "fallback"
    assert run.details["fallback_reason"] == "repair_limit_exhausted_downgrade"


def test_citation_critic_fails_without_answer_draft():
    result = citation_critic_agent(_state(answer_draft=None), config=_config())

    run = _run(result)
    assert run.status == "failed"
    assert result["repair_context"] is None
    assert run.details["stage"] == "input"
    assert result["errors"][0].agent == "citation_critic"
    assert result["errors"][0].agent_run_id == run.id


@patch("agents.citation_critic._call_llm")
def test_citation_critic_records_failure_on_llm_error(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (None, "provider unavailable")

    result = citation_critic_agent(_state(), config=_config(persistence))

    run = _run(result)
    assert run.status == "failed"
    assert run.output_ref == "state:errors"
    assert run.details["stage"] == "llm_call"
    assert "provider unavailable" in run.details["error"]
    assert persistence.list_runs() == [run]


@patch("agents.citation_critic._call_llm")
def test_citation_critic_rejects_invalid_review_shape(mock_call_llm):
    mock_call_llm.return_value = (
        _review_payload(confidence_adjustments=[]),
        None,
    )

    result = citation_critic_agent(_state(), config=_config())

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "normalization"
    assert "confidence_adjustments must be an object" in run.details["error"]


@patch("agents.citation_critic._call_llm")
def test_citation_critic_prompt_includes_answer_and_evidence(mock_call_llm):
    mock_call_llm.return_value = (_review_payload(), None)

    result = citation_critic_agent(_state(), config=_config())

    run = _run(result)
    user_content = mock_call_llm.call_args.args[0]
    assert "state:evidence_bundle" in run.input_refs
    assert "The method improves retrieval quality." in user_content
    assert '"chunk_id": "2310.06825:chunk:0"' in user_content


@patch("agents.citation_critic._call_llm")
def test_citation_critic_policy_override_reaches_agent(mock_call_llm):
    mock_call_llm.return_value = (_review_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=1,
        max_tokens=700,
        timeout_seconds=20,
        fallback_strategy="custom_critic_fallback",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"citation_critic": override}

    result = citation_critic_agent(_state(), config=config)

    run = _run(result)
    assert run.details["policy_applied"]["fallback_strategy"] == (
        "custom_critic_fallback"
    )
    assert mock_call_llm.call_args.kwargs["max_tokens"] == 700


@patch("agents.citation_critic._call_llm")
def test_citation_critic_policy_warning_records_limit_and_actual_calls(mock_call_llm):
    mock_call_llm.return_value = (_review_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=0,
        max_tokens=700,
        timeout_seconds=20,
        fallback_strategy="custom_critic_fallback",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"citation_critic": override}

    result = citation_critic_agent(_state(), config=config)

    run = _run(result)
    assert run.status == "completed"
    assert run.details["policy_warning"] == "exceeded_max_tool_calls"
    assert run.details["policy_max_tool_calls"] == 0
    assert run.details["actual_llm_call_count"] == 1
