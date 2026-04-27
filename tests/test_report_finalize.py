from agents.report_finalize import _current_url, report_finalize_node
from models.schemas import (
    BenchmarkResult,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)


def _base_state() -> dict:
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
        "text_by_page": {1: "page one"},
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
        "engineer_report": EngineerReport(
            executive_summary="Prototype now.",
            key_innovation="RL reasoning.",
            practical_implications="Good for reasoning workloads.",
            implementation_difficulty="moderate",
            recommended_action="prototype",
            action_reasoning="Open code and strong benchmarks.",
        ),
        "full_markdown_report": "# DeepSeek-R1",
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "completed",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {"extraction": 0.91},
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
        "messages": [],
        "errors": ["Benchmark Sonnet fallback used"],
        "cost_tracking": {},
    }


def test_current_url_uses_input_value_for_single_paper():
    state = _base_state()
    assert _current_url(state, 0) == "https://arxiv.org/abs/2501.12948"


def test_current_url_uses_batch_url_for_batch_mode():
    state = _base_state()
    state["batch_urls"] = [
        "https://arxiv.org/abs/2501.12948",
        "https://arxiv.org/abs/2305.14314",
    ]
    state["total_papers"] = 2

    assert _current_url(state, 1) == "https://arxiv.org/abs/2305.14314"


def test_report_finalize_packs_success_slot_and_resets_scratch_state():
    state = _base_state()

    result = report_finalize_node(state)

    assert result["processing_stage"] == "report_finalize"
    assert result["current_paper_index"] == 1
    assert result["errors"] == []
    assert result["metadata"] is None
    assert result["raw_text"] is None
    assert result["pdf_path"] is None
    assert result["text_by_page"] is None
    assert result["method_extraction"] is None
    assert result["benchmarks"] == []
    assert result["production_readiness"] is None
    assert result["engineer_report"] is None
    assert result["full_markdown_report"] is None
    assert result["ingestion_provenance"] is None
    assert result["confidence_scores"] == {}
    assert result["needs_human_review"] is False
    assert result["human_review_reason"] is None
    assert result["paper_failed"] is False
    assert result["paper_failure_reason"] is None
    assert result["failed_node"] is None

    papers = result["papers"]
    assert isinstance(papers, list)
    assert len(papers) == 1
    slot = papers[0]
    assert isinstance(slot, PaperSlot)
    assert slot.paper_index == 0
    assert slot.input_url == "https://arxiv.org/abs/2501.12948"
    assert slot.completed is True
    assert slot.markdown_report == "# DeepSeek-R1"
    assert slot.errors == ["Benchmark Sonnet fallback used"]


def test_report_finalize_requires_engineer_report():
    state = _base_state()
    state["engineer_report"] = None

    result = report_finalize_node(state)

    assert result["processing_stage"] == "failed"
    assert result["paper_failed"] is False
    assert result["failed_node"] == "report_finalize"
