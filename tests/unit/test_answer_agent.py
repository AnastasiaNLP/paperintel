import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.answer_agent import answer_agent
from models.agent_policies import AgentRuntimePolicy
from models.qa import RepairContext
from models.retrieval import (
    ChunkLocation,
    ChunkSearchResult,
    ChunkSource,
    EvidenceBundle,
    PaperChunk,
)


def _chunk(
    *,
    chunk_id: str = "2310.06825:chunk:0",
    text: str = "The method improves retrieval quality.",
) -> PaperChunk:
    return PaperChunk(
        id=chunk_id,
        paper_id="2310.06825",
        chunk_index=0,
        text=text,
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
        results=[
            ChunkSearchResult(
                chunk=chunk,
                score=0.9,
                rank=1,
                match_reason="semantic",
            )
        ],
    )


def _state(**overrides) -> dict:
    state = {
        "session_id": "session-1",
        "user_message": "What does the method improve?",
        "persona": "engineer",
        "evidence_bundle": _evidence(),
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


def _answer_payload(**overrides) -> str:
    payload = {
        "answer_text": "The method improves retrieval quality.",
        "citation_chunk_ids": ["2310.06825:chunk:0"],
        "confidence": 0.82,
        "limitations_noted": False,
        "insufficient_evidence": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


@patch("agents.answer_agent._call_llm")
def test_answer_agent_records_agent_run_on_success(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_answer_payload(), None)

    result = answer_agent(_state(), config=_config(persistence))

    run = _run(result)
    draft = result["answer_draft"]
    assert draft.answer_text == "The method improves retrieval quality."
    assert draft.citations[0].chunk_id == "2310.06825:chunk:0"
    assert draft.citations[0].page_start == 2
    assert run.agent_name == "answer_agent"
    assert run.model == "claude-sonnet-4-6"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:answer_draft"
    assert run.llm_call_count == 1
    assert run.details["policy_applied"]["fallback_strategy"] == (
        "insufficient_evidence_response"
    )
    assert persistence.list_runs() == [run]


@patch("agents.answer_agent._call_llm")
def test_answer_agent_uses_engineer_prompt_when_persona_is_engineer(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)

    answer_agent(_state(persona="engineer"), config=_config())

    system_prompt = mock_call_llm.call_args.args[0]
    assert "Persona: engineer." in system_prompt


@patch("agents.answer_agent._call_llm")
def test_answer_agent_uses_researcher_prompt_when_persona_is_researcher(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)

    answer_agent(_state(persona="researcher"), config=_config())

    system_prompt = mock_call_llm.call_args.args[0]
    assert "Persona: researcher." in system_prompt


@patch("agents.answer_agent._call_llm")
def test_answer_agent_uses_techlead_prompt_when_persona_is_techlead(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)

    answer_agent(_state(persona="techlead"), config=_config())

    system_prompt = mock_call_llm.call_args.args[0]
    assert "Persona: techlead." in system_prompt


def test_answer_agent_empty_evidence_returns_insufficient_evidence_fallback():
    evidence = EvidenceBundle(query="missing", results=[])

    result = answer_agent(_state(evidence_bundle=evidence), config=_config())

    run = _run(result)
    draft = result["answer_draft"]
    assert draft.insufficient_evidence is True
    assert draft.limitations_noted is True
    assert draft.confidence == 0.0
    assert draft.citations == []
    assert run.status == "fallback_used"
    assert run.termination_reason == "fallback"
    assert run.llm_call_count == 0
    assert run.details["fallback_reason"] == "no_evidence_available"


def test_answer_agent_fails_without_valid_persona():
    result = answer_agent(_state(persona=None), config=_config())

    run = _run(result)
    assert "answer_draft" not in result
    assert run.status == "failed"
    assert run.termination_reason == "error"
    assert run.details["stage"] == "persona"
    assert result["errors"][0].agent == "answer_agent"
    assert result["errors"][0].agent_run_id == run.id


@patch("agents.answer_agent._call_llm")
def test_answer_agent_records_failure_on_llm_error(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (None, "provider unavailable")

    result = answer_agent(_state(), config=_config(persistence))

    run = _run(result)
    assert run.status == "failed"
    assert run.output_ref == "state:errors"
    assert run.details["stage"] == "llm_call"
    assert "provider unavailable" in run.details["error"]
    assert persistence.list_runs() == [run]


@patch("agents.answer_agent._call_llm")
def test_answer_agent_rejects_unknown_citation_chunk_ids(mock_call_llm):
    mock_call_llm.return_value = (
        _answer_payload(citation_chunk_ids=["unknown-chunk"]),
        None,
    )

    result = answer_agent(_state(), config=_config())

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "normalization"
    assert "Unknown citation_chunk_ids" in run.details["error"]
    assert result["errors"][0].details["stage"] == "normalization"


@patch("agents.answer_agent._call_llm")
def test_answer_agent_prompt_includes_evidence_chunk_ids(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)

    result = answer_agent(_state(), config=_config())

    run = _run(result)
    user_content = mock_call_llm.call_args.args[1]
    assert "state:evidence_bundle" in run.input_refs
    assert '"chunk_id": "2310.06825:chunk:0"' in user_content
    assert "The method improves retrieval quality." in user_content


@patch("agents.answer_agent._call_llm")
def test_answer_agent_repair_context_propagates_iteration_and_instructions(
    mock_call_llm,
):
    mock_call_llm.return_value = (_answer_payload(), None)
    repair_context = RepairContext(
        original_run_id="run-1",
        target_agent="answer_agent",
        instructions=["Remove unsupported latency claim."],
        iteration=1,
        critic_review_id="review-1",
    )

    result = answer_agent(
        _state(repair_context=repair_context),
        config=_config(),
    )

    run = _run(result)
    draft = result["answer_draft"]
    user_content = mock_call_llm.call_args.args[1]
    assert draft.repair_iteration == 1
    assert result["repair_context"] is None
    assert run.details["repair_iteration"] == 1
    assert "Remove unsupported latency claim." in user_content
    assert f"repair_context:{repair_context.id}" in run.input_refs


@patch("agents.answer_agent._call_llm")
def test_answer_agent_policy_override_reaches_agent(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=500,
        timeout_seconds=15,
        fallback_strategy="custom_answer_fallback",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"answer_agent": override}

    result = answer_agent(_state(), config=config)

    run = _run(result)
    assert run.details["policy_applied"]["fallback_strategy"] == (
        "custom_answer_fallback"
    )
    assert mock_call_llm.call_args.kwargs["max_tokens"] == 500


@patch("agents.answer_agent._call_llm")
def test_answer_agent_policy_warning_records_limit_and_actual_calls(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=0,
        max_tokens=500,
        timeout_seconds=15,
        fallback_strategy="custom_answer_fallback",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"answer_agent": override}

    result = answer_agent(_state(), config=config)

    run = _run(result)
    assert run.status == "completed"
    assert run.details["policy_warning"] == "exceeded_max_tool_calls"
    assert run.details["policy_max_tool_calls"] == 0
    assert run.details["actual_llm_call_count"] == 1


@patch("agents.answer_agent._call_llm")
def test_answer_agent_distinct_run_ids_per_invocation(mock_call_llm):
    mock_call_llm.return_value = (_answer_payload(), None)

    first = answer_agent(_state(), config=_config())
    second = answer_agent(_state(), config=_config())

    assert _run(first).id != _run(second).id
