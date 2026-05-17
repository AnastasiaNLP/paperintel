import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.selection_advisor import selection_advisor_agent
from models.agent_policies import resolve_agent_policy
from models.discovery import SearchCandidate, SelectionAdvice


def _candidate(
    rank: int,
    *,
    candidate_id: str | None = None,
    title: str | None = None,
    year: int | None = 2024,
    arxiv_id: str | None = None,
    score: float = 1.0,
) -> SearchCandidate:
    arxiv_id = arxiv_id or f"2401.0000{rank}"
    return SearchCandidate(
        id=candidate_id or f"candidate-{rank}",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=rank,
        title=title or f"Agent Memory Paper {rank}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        year=year,
        arxiv_id=arxiv_id,
        score=score,
        reasons=["query terms in title", "recent paper"],
    )


def _candidates(count: int = 3) -> list[SearchCandidate]:
    return [_candidate(rank) for rank in range(1, count + 1)]


def _state(**overrides) -> dict:
    state = {
        "session_id": "session-1",
        "user_message": "Find papers about agent memory",
        "persona": "engineer",
        "discovery_topic": "agent memory",
        "search_candidates": _candidates(),
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


def _payload(**overrides) -> str:
    payload = {
        "response_text": (
            "I found several relevant papers. Start with 1 and 2 because they "
            "match agent memory most directly. Reply with numbers like 1, 3, 5."
        ),
        "recommended_candidate_ids": ["candidate-1", "candidate-2"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_formats_llm_advice(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    result = selection_advisor_agent(_state(), config=_config())

    advice = result["selection_advice"]
    run = _run(result)
    assert isinstance(advice, SelectionAdvice)
    assert advice.topic == "agent memory"
    assert advice.recommended_candidate_ids == ["candidate-1", "candidate-2"]
    assert advice.candidate_count == 3
    assert result["response_text"] == advice.response_text
    assert run.agent_name == "selection_advisor"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:selection_advice"
    assert run.llm_call_count == 1


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_uses_sonnet_model(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    result = selection_advisor_agent(_state(), config=_config())

    run = _run(result)
    assert run.model == "claude-sonnet-4-6"


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_includes_candidate_refs_in_input_refs(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    result = selection_advisor_agent(_state(), config=_config())

    run = _run(result)
    assert "state:search_candidates" in run.input_refs
    assert "search_candidate:candidate-1" in run.input_refs
    assert "search_candidate:candidate-2" in run.input_refs


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_prompt_includes_candidates(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    selection_advisor_agent(_state(), config=_config())

    user_content = mock_call_llm.call_args.args[0]
    assert '"topic": "agent memory"' in user_content
    assert '"persona": "engineer"' in user_content
    assert '"id": "candidate-1"' in user_content
    assert "Agent Memory Paper 1" in user_content


def test_selection_advisor_requires_discovery_topic():
    result = selection_advisor_agent(
        _state(discovery_topic="", user_message=""),
        config=_config(),
    )

    run = _run(result)
    assert run.status == "failed"
    assert run.output_ref == "state:errors"
    assert result["errors"][0].code == "SELECTION_ADVISOR_FAILED"


def test_selection_advisor_falls_back_without_candidates():
    result = selection_advisor_agent(_state(search_candidates=[]), config=_config())

    advice = result["selection_advice"]
    run = _run(result)
    assert advice.recommended_candidate_ids == []
    assert "did not find candidate papers" in advice.response_text
    assert run.status == "fallback_used"
    assert run.llm_call_count == 0
    assert run.details["fallback_reason"] == "no_candidates_available"


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_drops_unknown_recommended_ids(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(recommended_candidate_ids=["candidate-1", "ghost"]),
        None,
    )

    result = selection_advisor_agent(_state(), config=_config())

    advice = result["selection_advice"]
    assert advice.recommended_candidate_ids == ["candidate-1"]


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_falls_back_when_no_valid_recommendations(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(recommended_candidate_ids=["ghost"]),
        None,
    )

    result = selection_advisor_agent(_state(), config=_config())

    advice = result["selection_advice"]
    run = _run(result)
    assert advice.recommended_candidate_ids == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert run.status == "completed"


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_falls_back_on_llm_error(mock_call_llm):
    mock_call_llm.return_value = (None, "provider timeout")

    result = selection_advisor_agent(_state(), config=_config())

    advice = result["selection_advice"]
    run = _run(result)
    assert advice.recommended_candidate_ids == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert "Reply with the numbers" in advice.response_text
    assert run.status == "fallback_used"
    assert run.details["fallback_reason"] == "provider timeout"


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_falls_back_on_invalid_json(mock_call_llm):
    mock_call_llm.return_value = ("not-json", None)

    result = selection_advisor_agent(_state(), config=_config())

    run = _run(result)
    assert run.status == "fallback_used"
    assert "JSON parse error" in run.details["fallback_reason"]


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_fallback_lists_top_five(mock_call_llm):
    mock_call_llm.return_value = ("not-json", None)

    result = selection_advisor_agent(_state(search_candidates=_candidates(7)), config=_config())

    text = result["selection_advice"].response_text
    assert "1. 1 | Agent Memory Paper 1" in text
    assert "5. 5 | Agent Memory Paper 5" in text
    assert "6. 6 | Agent Memory Paper 6" not in text


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_persists_run_on_success(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_payload(), None)

    result = selection_advisor_agent(_state(), config=_config(persistence))

    assert persistence.list_runs() == result["agent_runs"]


@patch("agents.selection_advisor._call_llm")
def test_selection_advisor_persists_run_on_fallback(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = ("not-json", None)

    result = selection_advisor_agent(_state(), config=_config(persistence))

    assert persistence.list_runs() == result["agent_runs"]


def test_selection_advisor_policy_registered():
    policy = resolve_agent_policy("selection_advisor")

    assert policy.max_iterations == 1
    assert policy.max_tool_calls == 1
    assert policy.max_tokens == 2500
    assert policy.timeout_seconds == 45
    assert policy.fallback_strategy == "deterministic_shortlist"
