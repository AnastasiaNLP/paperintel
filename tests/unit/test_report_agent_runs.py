import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.report import report_agent
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy
from models.schemas import (
    BenchmarkResult,
    MethodExtraction,
    PaperMetadata,
    ProductionReadiness,
)


def _claims(**overrides) -> str:
    data = {
        "executive_summary": "Use this for reasoning prototypes.",
        "key_innovation": "RL-first reasoning pipeline.",
        "practical_implications": "Useful for math and coding workloads.",
        "implementation_difficulty": "moderate",
        "recommended_action": "prototype",
        "action_reasoning": "Open code and strong benchmark coverage.",
    }
    data.update(overrides)
    return json.dumps(data)


def _state() -> dict:
    return {
        "input_type": "url",
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "papers": [],
        "metadata": PaperMetadata(
            title="DeepSeek-R1",
            authors=["DeepSeek-AI"],
            arxiv_id="2501.12948",
            published_date="2025-01-22",
            abstract="Reasoning model.",
            categories=["cs.AI"],
            citation_count=1,
        ),
        "raw_text": "paper text",
        "pdf_path": "/tmp/2501.12948.pdf",
        "text_by_page": {1: "page text"},
        "method_extraction": MethodExtraction(
            method_name="DeepSeek-R1",
            description="Reasoning-focused LLM.",
            novelty_claim="RL-first reasoning.",
            key_components=["GRPO"],
            compared_to=["o1"],
            limitations_stated=["cost"],
        ),
        "benchmarks": [
            BenchmarkResult(
                task="MATH-500",
                metric="pass@1",
                value=97.3,
                conditions="reasoning evaluation",
            )
        ],
        "production_readiness": ProductionReadiness(
            has_open_code=True,
            code_url="https://github.com/deepseek-ai/DeepSeek-R1",
            huggingface_model="deepseek-ai/DeepSeek-R1",
            framework_integrations=["vLLM"],
            min_gpu_requirement=None,
            estimated_inference_cost=None,
            dependencies=["torch"],
            maturity_level="experimental",
            maturity_reasoning="Verified model and code.",
        ),
        "ingestion_provenance": {
            "text_source": "pdf",
            "metadata_source": "arxiv",
            "enrichment_status": "s2_ok",
            "arxiv_id_found": True,
        },
        "comparison_markdown": None,
        "comparison_report": None,
        "engineer_report": None,
        "full_markdown_report": None,
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "report",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
        "messages": [],
        "errors": [],
        "agent_runs": [],
        "cost_tracking": {},
    }


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


@patch("agents.report._call_llm")
def test_report_success_creates_completed_agent_run(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_claims(), None)

    result = report_agent(_state(), config=_config(persistence))

    run = _run(result)
    assert result["processing_stage"] == "completed"
    assert run.agent_name == "report"
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "state:report"
    assert run.iteration_count == 1
    assert run.llm_call_count == 1
    assert run.session_id == "session-1"
    assert run.job_id == "job-1"
    assert run.details["parse_repair_attempted"] is False
    assert run.details["normalized"] is False
    assert run.details["policy_applied"]["fallback_strategy"] == "repair_on_invalid_json"
    assert persistence.list_runs() == [run]


@patch("agents.report._call_llm")
def test_report_without_persistence_config_uses_noop(mock_call_llm):
    mock_call_llm.return_value = (_claims(), None)

    result = report_agent(_state(), config=None)

    run = _run(result)
    assert run.status == "completed"
    assert run.termination_reason == "success"


@patch("agents.report._call_llm_repair")
@patch("agents.report._call_llm")
def test_report_repair_success_counts_llm_calls_not_iterations(
    mock_call_llm,
    mock_repair,
):
    mock_call_llm.return_value = ("not json", None)
    mock_repair.return_value = (_claims(), None)

    result = report_agent(_state(), config=_config())

    run = _run(result)
    assert result["processing_stage"] == "completed"
    assert run.status == "completed"
    assert run.iteration_count == 1
    assert run.llm_call_count == 2
    assert run.details["parse_repair_attempted"] is True


@patch("agents.report._call_llm_repair")
@patch("agents.report._call_llm")
def test_report_policy_warning_records_limit_and_actual_counts(
    mock_call_llm,
    mock_repair,
):
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=1200,
        timeout_seconds=30,
        fallback_strategy="no_repair",
    )
    config = _config()
    config["configurable"]["agent_policy_overrides"] = {"report": override}
    mock_call_llm.return_value = ("not json", None)
    mock_repair.return_value = (_claims(), None)

    result = report_agent(_state(), config=config)

    run = _run(result)
    assert run.status == "completed"
    assert run.llm_call_count == 2
    assert run.details["policy_warning"] == "exceeded_max_tool_calls"
    assert run.details["policy_max_tool_calls"] == 1
    assert run.details["actual_llm_call_count"] == 2
    assert run.details["policy_applied"]["fallback_strategy"] == "no_repair"


@patch("agents.report._call_llm")
def test_report_llm_error_creates_failed_agent_run(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (None, "provider unavailable")

    result = report_agent(_state(), config=_config(persistence))

    run = _run(result)
    assert result["processing_stage"] == "failed"
    assert run.status == "failed"
    assert run.termination_reason == "error"
    assert run.output_ref == "state:errors"
    assert run.llm_call_count == 1
    assert run.details["stage"] == "llm_call"
    assert "provider unavailable" in run.details["error"]
    assert persistence.list_runs() == [run]


@patch("agents.report._call_llm_repair")
@patch("agents.report._call_llm")
def test_report_repair_failure_creates_failed_agent_run(mock_call_llm, mock_repair):
    mock_call_llm.return_value = ("not json", None)
    mock_repair.return_value = (None, "repair unavailable")

    result = report_agent(_state(), config=_config())

    run = _run(result)
    assert result["processing_stage"] == "failed"
    assert run.status == "failed"
    assert run.termination_reason == "error"
    assert run.iteration_count == 1
    assert run.llm_call_count == 2
    assert run.details["stage"] == "repair"
    assert run.details["parse_repair_attempted"] is True


@patch("agents.report._call_llm")
def test_report_repeated_runs_have_distinct_agent_run_ids(mock_call_llm):
    mock_call_llm.return_value = (_claims(), None)

    first = report_agent(_state(), config=_config())
    second = report_agent(_state(), config=_config())

    assert _run(first).id != _run(second).id
    assert _run(first).output_ref == _run(second).output_ref == "state:report"


@patch("agents.report._call_llm")
def test_report_run_appends_to_existing_state_via_reducer_contract(mock_call_llm):
    existing = AgentRun(agent_name="ingestion")
    mock_call_llm.return_value = (_claims(), None)

    result = report_agent({**_state(), "agent_runs": [existing]}, config=_config())

    assert result["agent_runs"] != [existing]
    assert result["agent_runs"][0].agent_name == "report"
