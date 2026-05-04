from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.evidence_critic import evidence_critic_agent
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy
from models.errors import error_message
from models.schemas import BenchmarkResult, EngineerReport, ProductionReadiness


def _report(action: str = "implement_now", difficulty: str = "moderate") -> EngineerReport:
    return EngineerReport(
        executive_summary="Summary",
        key_innovation="Innovation",
        practical_implications="Implications",
        implementation_difficulty=difficulty,
        recommended_action=action,
        action_reasoning="Initial reasoning.",
    )


def _readiness(maturity: str = "experimental") -> ProductionReadiness:
    return ProductionReadiness(
        has_open_code=True,
        code_url="https://github.com/example/repo",
        huggingface_model=None,
        framework_integrations=[],
        min_gpu_requirement=None,
        estimated_inference_cost=None,
        dependencies=[],
        maturity_level=maturity,
        maturity_reasoning="Reasoning",
    )


def _config(persistence=None) -> dict:
    return {
        "configurable": {
            "session_id": "session-1",
            "job_id": "job-1",
            "agent_run_persistence": persistence or InMemoryAgentRunPersistence(),
        }
    }


def _run(result: dict) -> AgentRun:
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


def test_evidence_critic_accepts_supported_prototype_report():
    persistence = InMemoryAgentRunPersistence()
    report_run = AgentRun(agent_name="report")
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="prototype"),
            "benchmarks": [BenchmarkResult(task="MMLU", metric="accuracy", value=80.0)],
            "production_readiness": _readiness("experimental"),
            "agent_runs": [report_run],
        },
        config=_config(persistence),
    )

    run = _run(result)
    assert result.keys() == {"agent_runs"}
    assert run.agent_name == "evidence_critic"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:report"
    assert run.input_refs == ["state:report", report_run.id]
    assert run.iteration_count == 1
    assert run.llm_call_count == 0
    assert run.details["reviewed"] is True
    assert run.details["changed"] is False
    assert run.details["policy_applied"]["fallback_strategy"] == "skip_review_on_no_report"
    assert persistence.list_runs() == [run]


def test_evidence_critic_downgrades_implement_now_without_benchmarks():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="implement_now"),
            "benchmarks": [],
            "production_readiness": _readiness("production_ready"),
        },
        config=_config(),
    )

    updated = result["engineer_report"]
    run = _run(result)
    assert updated.recommended_action == "prototype"
    assert "no benchmarks were extracted" in updated.action_reasoning
    assert "implement_now without benchmark evidence" in error_message(result["errors"][0])
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.details["changed"] is True
    assert run.details["warnings_count"] == 1


def test_evidence_critic_downgrades_when_readiness_missing():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="prototype"),
            "benchmarks": [BenchmarkResult(task="MMLU", metric="accuracy", value=80.0)],
            "production_readiness": None,
        },
        config=_config(),
    )

    updated = result["engineer_report"]
    assert updated.recommended_action == "watch"
    assert updated.implementation_difficulty == "research_only"
    assert "production readiness evidence is unavailable" in updated.action_reasoning


def test_evidence_critic_downgrades_research_only_maturity():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="prototype"),
            "benchmarks": [BenchmarkResult(task="MMLU", metric="accuracy", value=80.0)],
            "production_readiness": _readiness("research_only"),
        },
        config=_config(),
    )

    updated = result["engineer_report"]
    assert updated.recommended_action == "watch"
    assert updated.implementation_difficulty == "research_only"
    assert "maturity is research_only" in updated.action_reasoning


def test_evidence_critic_noops_without_report():
    result = evidence_critic_agent(
        {"engineer_report": None, "agent_runs": []},
        config=_config(),
    )

    run = _run(result)
    assert result.keys() == {"agent_runs"}
    assert run.status == "completed"
    assert run.termination_reason == "skipped"
    assert run.output_ref == "state:report"
    assert run.input_refs == ["state:report"]
    assert run.details["reason"] == "no_report_to_review"
    assert run.details["fallback_used"] is True
    assert run.details["fallback_reason"] == "no_report_to_review"
    assert (
        run.details["policy_applied"]["fallback_strategy"]
        == "skip_review_on_no_report"
    )


def test_evidence_critic_policy_override_reaches_agent():
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=0,
        max_tokens=None,
        timeout_seconds=None,
        fallback_strategy="custom_skip_strategy",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"evidence_critic": override}

    result = evidence_critic_agent(
        {"engineer_report": None, "agent_runs": []},
        config=config,
    )

    run = _run(result)
    assert run.termination_reason == "skipped"
    assert run.details["policy_applied"]["fallback_strategy"] == "custom_skip_strategy"
