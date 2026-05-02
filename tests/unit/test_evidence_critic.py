from agents.evidence_critic import evidence_critic_agent
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


def test_evidence_critic_accepts_supported_prototype_report():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="prototype"),
            "benchmarks": [BenchmarkResult(task="MMLU", metric="accuracy", value=80.0)],
            "production_readiness": _readiness("experimental"),
        }
    )

    assert result == {}


def test_evidence_critic_downgrades_implement_now_without_benchmarks():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="implement_now"),
            "benchmarks": [],
            "production_readiness": _readiness("production_ready"),
        }
    )

    updated = result["engineer_report"]
    assert updated.recommended_action == "prototype"
    assert "no benchmarks were extracted" in updated.action_reasoning
    assert "implement_now without benchmark evidence" in error_message(result["errors"][0])


def test_evidence_critic_downgrades_when_readiness_missing():
    result = evidence_critic_agent(
        {
            "engineer_report": _report(action="prototype"),
            "benchmarks": [BenchmarkResult(task="MMLU", metric="accuracy", value=80.0)],
            "production_readiness": None,
        }
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
        }
    )

    updated = result["engineer_report"]
    assert updated.recommended_action == "watch"
    assert updated.implementation_difficulty == "research_only"
    assert "maturity is research_only" in updated.action_reasoning


def test_evidence_critic_noops_without_report():
    assert evidence_critic_agent({"engineer_report": None}) == {}
