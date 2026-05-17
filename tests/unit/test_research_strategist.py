import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.research_strategist import research_strategist_agent
from models.agent_policies import resolve_agent_policy
from models.discovery import DiscoveryPlan


def _config(persistence=None) -> dict:
    return {
        "configurable": {
            "session_id": "session-1",
            "job_id": "job-1",
            "agent_run_persistence": persistence or InMemoryAgentRunPersistence(),
        }
    }


def _state(message: str = "Find papers about RAG citation grounding") -> dict:
    return {
        "session_id": "session-1",
        "user_message": message,
        "persona": "engineer",
    }


def _payload(**overrides) -> str:
    payload = {
        "topic": "RAG citation grounding",
        "queries": [
            {
                "query": "retrieval augmented generation citation grounding",
                "max_results": 10,
                "source": "arxiv",
            },
            {
                "query": "faithful citations large language models retrieval",
                "max_results": 8,
                "source": "arxiv",
            },
        ],
        "reasoning": "The topic needs grounding and citation-focused search terms.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_parses_valid_plan(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    result = research_strategist_agent(_state(), config=_config())

    plan = result["discovery_plan"]
    run = _run(result)
    assert isinstance(plan, DiscoveryPlan)
    assert result["discovery_topic"] == "RAG citation grounding"
    assert [query.query for query in plan.queries] == [
        "retrieval augmented generation citation grounding",
        "faithful citations large language models retrieval",
    ]
    assert run.agent_name == "research_strategist"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:discovery_plan"
    assert run.llm_call_count == 1


@patch("agents.research_strategist._call_llm")
def test_research_strategist_uses_haiku_model(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    result = research_strategist_agent(_state(), config=_config())

    run = _run(result)
    assert run.model == "claude-haiku-4-5-20251001"


@patch("agents.research_strategist._call_llm")
def test_research_strategist_prompt_includes_message_and_persona(mock_call_llm):
    mock_call_llm.return_value = (_payload(), None)

    research_strategist_agent(_state(), config=_config())

    user_content = mock_call_llm.call_args.args[0]
    assert '"message": "Find papers about RAG citation grounding"' in user_content
    assert '"persona": "engineer"' in user_content


@patch("agents.research_strategist._call_llm")
def test_research_strategist_limits_queries_to_four(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(
            queries=[
                {"query": "query one", "max_results": 10, "source": "arxiv"},
                {"query": "query two", "max_results": 10, "source": "arxiv"},
                {"query": "query three", "max_results": 10, "source": "arxiv"},
                {"query": "query four", "max_results": 10, "source": "arxiv"},
                {"query": "query five", "max_results": 10, "source": "arxiv"},
            ]
        ),
        None,
    )

    result = research_strategist_agent(_state(), config=_config())

    assert [query.query for query in result["discovery_plan"].queries] == [
        "query one",
        "query two",
        "query three",
        "query four",
    ]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_normalizes_query_max_results(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(
            queries=[
                {"query": "too many", "max_results": 100, "source": "arxiv"},
                {"query": "too few", "max_results": 0, "source": "arxiv"},
                {"query": "not an int", "max_results": "bad", "source": "arxiv"},
            ]
        ),
        None,
    )

    result = research_strategist_agent(_state(), config=_config())

    assert [query.max_results for query in result["discovery_plan"].queries] == [
        10,
        1,
        10,
    ]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_forces_arxiv_source(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(
            queries=[
                {"query": "citation grounding", "max_results": 10, "source": "web"},
            ]
        ),
        None,
    )

    result = research_strategist_agent(_state(), config=_config())

    assert result["discovery_plan"].queries[0].source == "arxiv"


@patch("agents.research_strategist._call_llm")
def test_research_strategist_deduplicates_queries(mock_call_llm):
    mock_call_llm.return_value = (
        _payload(
            queries=[
                {"query": "citation grounding", "max_results": 10, "source": "arxiv"},
                {"query": "Citation Grounding", "max_results": 8, "source": "arxiv"},
            ]
        ),
        None,
    )

    result = research_strategist_agent(_state(), config=_config())

    assert [query.query for query in result["discovery_plan"].queries] == [
        "citation grounding"
    ]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_uses_fallback_topic_for_blank_llm_topic(mock_call_llm):
    mock_call_llm.return_value = (_payload(topic="   "), None)

    result = research_strategist_agent(_state(), config=_config())

    run = _run(result)
    assert result["discovery_topic"] == "Find papers about RAG citation grounding"
    assert result["discovery_plan"].topic == "Find papers about RAG citation grounding"
    assert run.status == "completed"
    assert run.termination_reason == "success"


@patch("agents.research_strategist._call_llm")
def test_research_strategist_falls_back_on_invalid_json(mock_call_llm):
    mock_call_llm.return_value = ("not-json", None)

    result = research_strategist_agent(_state(), config=_config())

    plan = result["discovery_plan"]
    run = _run(result)
    assert plan.topic == "Find papers about RAG citation grounding"
    assert plan.queries[0].query == "Find papers about RAG citation grounding"
    assert run.status == "fallback_used"
    assert run.termination_reason == "fallback"
    assert run.details["fallback_used"] is True
    assert "JSON parse error" in run.details["fallback_reason"]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_falls_back_on_empty_queries(mock_call_llm):
    mock_call_llm.return_value = (_payload(queries=[]), None)

    result = research_strategist_agent(_state(), config=_config())

    plan = result["discovery_plan"]
    run = _run(result)
    assert plan.queries[0].query == "Find papers about RAG citation grounding"
    assert run.status == "fallback_used"
    assert "at least one valid query" in run.details["fallback_reason"]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_falls_back_on_llm_error(mock_call_llm):
    mock_call_llm.return_value = (None, "provider timeout")

    result = research_strategist_agent(_state(), config=_config())

    run = _run(result)
    assert result["discovery_plan"].queries[0].max_results == 10
    assert run.status == "fallback_used"
    assert run.details["fallback_reason"] == "provider timeout"


def test_research_strategist_requires_user_message():
    result = research_strategist_agent(_state("   "), config=_config())

    run = _run(result)
    assert "errors" in result
    assert run.status == "failed"
    assert run.output_ref == "state:errors"
    assert result["errors"][0].code == "RESEARCH_STRATEGIST_FAILED"


@patch("agents.research_strategist._call_llm")
def test_research_strategist_persists_run_on_success(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_payload(), None)

    result = research_strategist_agent(_state(), config=_config(persistence))

    assert persistence.list_runs() == result["agent_runs"]


@patch("agents.research_strategist._call_llm")
def test_research_strategist_persists_run_on_fallback(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = ("not-json", None)

    result = research_strategist_agent(_state(), config=_config(persistence))

    assert persistence.list_runs() == result["agent_runs"]


def test_research_strategist_policy_registered():
    policy = resolve_agent_policy("research_strategist")

    assert policy.max_iterations == 1
    assert policy.max_tool_calls == 1
    assert policy.max_tokens == 1500
    assert policy.timeout_seconds == 20
    assert policy.fallback_strategy == "single_query_fallback"
