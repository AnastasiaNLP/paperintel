import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.intent_router import intent_router_agent
from api.in_memory_session_store import InMemorySessionStore
from models.agent_policies import AgentRuntimePolicy


def _store(
    *,
    persona: str = "engineer",
    active_paper_ids: list[str] | None = None,
) -> tuple[InMemorySessionStore, str]:
    store = InMemorySessionStore()
    session = store.create_session(persona=persona)
    stored = store._sessions[session.id]
    stored.active_paper_ids = (
        ["2310.06825"] if active_paper_ids is None else active_paper_ids
    )
    return store, session.id


def _config(session_id: str, store: InMemorySessionStore, persistence=None) -> dict:
    return {
        "configurable": {
            "session_id": session_id,
            "job_id": "job-1",
            "session_store": store,
            "agent_run_persistence": persistence or InMemoryAgentRunPersistence(),
        }
    }


def _state(session_id: str, message: str = "What is the loss?") -> dict:
    return {
        "session_id": session_id,
        "user_message": message,
    }


def _payload(**overrides) -> str:
    payload = {
        "intent": "qa_factual",
        "referenced_paper_ids": ["2310.06825"],
        "ambiguous": False,
        "clarification_question": None,
        "confidence": 0.86,
        "reasoning": "The user asks a factual question about the active paper.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_qa_factual(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(intent="qa_factual"), None)

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    run = _run(result)
    assert result["intent"] == "qa_factual"
    assert result["referenced_paper_ids"] == ["2310.06825"]
    assert result["needs_clarification"] is False
    assert run.agent_name == "intent_router"
    assert run.model == "claude-haiku-4-5-20251001"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:intent_resolution"
    assert run.llm_call_count == 1
    assert run.details["policy_applied"]["fallback_strategy"] == "ask_clarification"


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_qa_math(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(intent="qa_math"), None)

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    assert result["intent"] == "qa_math"


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_qa_comparison(mock_call_llm):
    store, session_id = _store(active_paper_ids=["2310.06825", "2401.12345"])
    mock_call_llm.return_value = (
        _payload(
            intent="qa_comparison",
            referenced_paper_ids=["2310.06825", "2401.12345"],
        ),
        None,
    )

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    assert result["intent"] == "qa_comparison"
    assert result["referenced_paper_ids"] == ["2310.06825", "2401.12345"]


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_qa_followup(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(intent="qa_followup"), None)

    result = intent_router_agent(_state(session_id, "What about latency?"), config=_config(session_id, store))

    assert result["intent"] == "qa_followup"


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_discover(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (
        _payload(intent="discover", referenced_paper_ids=[]),
        None,
    )

    result = intent_router_agent(
        _state(session_id, "Find papers about memory for agents"),
        config=_config(session_id, store),
    )

    assert result["intent"] == "discover"
    assert result["referenced_paper_ids"] == []


@patch("agents.intent_router._call_llm")
def test_intent_router_classifies_analyze_paper(mock_call_llm):
    store, session_id = _store(active_paper_ids=[])
    mock_call_llm.return_value = (
        _payload(intent="analyze_paper", referenced_paper_ids=[]),
        None,
    )

    result = intent_router_agent(
        _state(session_id, "Analyze https://arxiv.org/abs/2310.06825"),
        config=_config(session_id, store),
    )

    assert result["intent"] == "analyze_paper"
    assert result["needs_clarification"] is False


@patch("agents.intent_router._call_llm")
def test_intent_router_ambiguous_returns_clarification(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (
        _payload(
            intent="clarification_needed",
            referenced_paper_ids=[],
            ambiguous=True,
            clarification_question="Which paper should I use?",
            confidence=0.4,
        ),
        None,
    )

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    assert result["intent"] == "clarification_needed"
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Which paper should I use?"


@patch("agents.intent_router._call_llm")
def test_intent_router_propagates_persona_from_session(mock_call_llm):
    store, session_id = _store(persona="researcher")
    mock_call_llm.return_value = (_payload(), None)

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    assert result["persona"] == "researcher"


@patch("agents.intent_router._call_llm")
def test_intent_router_includes_recent_turns_in_prompt_payload(mock_call_llm):
    store, session_id = _store()
    store.append_turn(
        session_id,
        role="user",
        content="Analyze https://arxiv.org/abs/2310.06825",
        intent="analyze_paper",
    )
    store.append_turn(
        session_id,
        role="assistant",
        content="Analyzed paper 2310.06825.",
        referenced_paper_ids=["2310.06825"],
    )
    mock_call_llm.return_value = (_payload(), None)

    intent_router_agent(_state(session_id), config=_config(session_id, store))

    user_content = mock_call_llm.call_args.args[0]
    assert '"active_paper_ids": [' in user_content
    assert '"persona": "engineer"' in user_content
    assert "Analyze https://arxiv.org/abs/2310.06825" in user_content
    assert "Analyzed paper 2310.06825." in user_content


@patch("agents.intent_router._call_llm")
def test_intent_router_validates_unknown_referenced_paper_ids(mock_call_llm):
    store, session_id = _store(active_paper_ids=["2310.06825", "2401.12345"])
    mock_call_llm.return_value = (
        _payload(referenced_paper_ids=["2310.06825", "ghost-paper-xyz"]),
        None,
    )

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    resolution = result["intent_resolution"]
    assert result["intent"] == "clarification_needed"
    assert result["referenced_paper_ids"] == []
    assert result["needs_clarification"] is True
    assert "Available paper ids" in result["clarification_question"]
    assert "ghost-paper-xyz" in (resolution.reasoning or "")


@patch("agents.intent_router._call_llm")
def test_intent_router_forces_clarification_for_qa_without_active_papers(mock_call_llm):
    store, session_id = _store(active_paper_ids=[])
    mock_call_llm.return_value = (
        _payload(intent="qa_factual", referenced_paper_ids=[]),
        None,
    )

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    assert result["intent"] == "clarification_needed"
    assert result["needs_clarification"] is True
    assert "do not have an analyzed paper" in result["clarification_question"]


@patch("agents.intent_router._call_llm")
def test_intent_router_records_agent_run_on_success(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(), None)

    result = intent_router_agent(
        _state(session_id),
        config=_config(session_id, store, persistence),
    )

    run = _run(result)
    assert run.input_refs == [
        "state:user_message",
        f"session:{session_id}",
        "turns:recent:10",
    ]
    assert persistence.list_runs() == [run]


@patch("agents.intent_router._call_llm")
def test_intent_router_fallbacks_to_clarification_on_llm_error(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (None, "provider unavailable")

    result = intent_router_agent(_state(session_id), config=_config(session_id, store))

    run = _run(result)
    assert result["intent"] == "clarification_needed"
    assert result["needs_clarification"] is True
    assert run.status == "fallback_used"
    assert run.termination_reason == "fallback"
    assert run.details["fallback_reason"] == "llm_error"


def test_intent_router_fails_without_session_store():
    result = intent_router_agent(
        {"session_id": "session-1", "user_message": "What is the loss?"},
        config={"configurable": {"session_id": "session-1"}},
    )

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "session_context"
    assert result["errors"][0].agent == "intent_router"


@patch("agents.intent_router._call_llm")
def test_intent_router_policy_override_reaches_agent(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=600,
        timeout_seconds=10,
        fallback_strategy="custom_clarification",
    )
    config = _config(session_id, store)
    config["configurable"]["agent_policy_overrides"] = {"intent_router": override}

    result = intent_router_agent(_state(session_id), config=config)

    run = _run(result)
    assert run.details["policy_applied"]["fallback_strategy"] == "custom_clarification"
    assert mock_call_llm.call_args.kwargs["max_tokens"] == 600


@patch("agents.intent_router._call_llm")
def test_intent_router_policy_warning_records_limit_and_actual_calls(mock_call_llm):
    store, session_id = _store()
    mock_call_llm.return_value = (_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=0,
        max_tokens=600,
        timeout_seconds=10,
        fallback_strategy="custom_clarification",
    )
    config = _config(session_id, store)
    config["configurable"]["agent_policy_overrides"] = {"intent_router": override}

    result = intent_router_agent(_state(session_id), config=config)

    run = _run(result)
    assert run.status == "completed"
    assert run.details["policy_warning"] == "exceeded_max_tool_calls"
    assert run.details["policy_max_tool_calls"] == 0
    assert run.details["actual_llm_call_count"] == 1
