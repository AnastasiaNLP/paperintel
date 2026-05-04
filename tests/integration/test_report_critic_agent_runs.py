import json
import uuid
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.evidence_critic import evidence_critic_agent
from agents.report import report_agent
from models.agent_runs import AgentRun
from models.schemas import (
    BenchmarkResult,
    MethodExtraction,
    PaperMetadata,
    ProductionReadiness,
)
from models.state import PaperIntelState


def _claims() -> str:
    return json.dumps(
        {
            "executive_summary": "Use this for reasoning prototypes.",
            "key_innovation": "RL-first reasoning pipeline.",
            "practical_implications": "Useful for math and coding workloads.",
            "implementation_difficulty": "moderate",
            "recommended_action": "prototype",
            "action_reasoning": "Open code and strong benchmark coverage.",
        }
    )


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


def _build_report_critic_graph(*, checkpointer=None, interrupt_after=None):
    graph = StateGraph(PaperIntelState)
    graph.add_node("report", report_agent)
    graph.add_node("evidence_critic", evidence_critic_agent)
    graph.set_entry_point("report")
    graph.add_edge("report", "evidence_critic")
    graph.add_edge("evidence_critic", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("models.schemas", "PaperMetadata"),
            ("models.schemas", "MethodExtraction"),
            ("models.schemas", "BenchmarkResult"),
            ("models.schemas", "ProductionReadiness"),
            ("models.schemas", "EngineerReport"),
            ("models.schemas", "PaperSlot"),
            ("models.schemas", "ComparisonMatrixRow"),
            ("models.schemas", "ConstraintRecommendation"),
            ("models.schemas", "ComparisonReport"),
            ("models.agent_runs", "AgentRun"),
            ("models.errors", "StructuredError"),
        ],
    )


def _config(persistence: InMemoryAgentRunPersistence) -> dict:
    return {
        "configurable": {
            "thread_id": f"report-critic-{uuid.uuid4()}",
            "session_id": "session-1",
            "job_id": "job-1",
            "agent_run_persistence": persistence,
        }
    }


def _assert_report_critic_runs(runs: list[AgentRun]) -> None:
    assert len(runs) == 2
    report_run, critic_run = runs

    assert report_run.agent_name == "report"
    assert report_run.status == "completed"
    assert report_run.termination_reason == "success"
    assert report_run.output_ref == "state:report"
    assert report_run.details["parse_repair_attempted"] is False
    assert report_run.details["policy_applied"]["fallback_strategy"] == (
        "repair_on_invalid_json"
    )

    assert critic_run.agent_name == "evidence_critic"
    assert critic_run.status == "completed"
    assert critic_run.termination_reason == "success"
    assert critic_run.output_ref == "state:report"
    assert critic_run.input_refs == ["state:report", report_run.id]
    assert critic_run.details["reviewed"] is True
    assert critic_run.details["changed"] is False
    assert critic_run.details["policy_applied"]["fallback_strategy"] == (
        "skip_review_on_no_report"
    )


@patch("agents.report._call_llm")
def test_report_to_evidence_critic_accumulates_agent_runs_and_persistence(
    mock_call_llm,
):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_claims(), None)
    app = _build_report_critic_graph()

    result = app.invoke(_state(), config=_config(persistence))

    _assert_report_critic_runs(result["agent_runs"])
    assert persistence.list_runs() == result["agent_runs"]


@patch("agents.report._call_llm")
def test_report_critic_agent_runs_survive_checkpoint_resume(mock_call_llm):
    persistence = InMemoryAgentRunPersistence()
    mock_call_llm.return_value = (_claims(), None)
    checkpointer = MemorySaver(serde=_serde())
    app = _build_report_critic_graph(
        checkpointer=checkpointer,
        interrupt_after=["report"],
    )
    config = _config(persistence)

    interrupted = app.invoke(_state(), config=config)
    assert len(interrupted["agent_runs"]) == 1
    assert interrupted["agent_runs"][0].agent_name == "report"
    assert app.get_state(config).next == ("evidence_critic",)

    resumed = app.invoke(None, config=config)

    _assert_report_critic_runs(resumed["agent_runs"])
    _assert_report_critic_runs(app.get_state(config).values["agent_runs"])
    assert persistence.list_runs() == resumed["agent_runs"]
