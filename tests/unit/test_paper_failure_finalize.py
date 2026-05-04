from agents.paper_failure_finalize import paper_failure_finalize_node
from models.schemas import BenchmarkResult, MethodExtraction, PaperMetadata, PaperSlot


def _failed_state() -> dict:
    return {
        "input_type": "url",
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "papers": [],
        "metadata": None,
        "raw_text": "paper text",
        "pdf_path": "/tmp/2501.12948.pdf",
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
        "ingestion_provenance": None,
        "comparison_markdown": None,
        "comparison_report": None,
        "engineer_report": None,
        "full_markdown_report": None,
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "paper_failure_finalize",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "paper_failed": True,
        "paper_failure_reason": "PDF parse failed",
        "failed_node": "ingestion",
        "messages": [],
        "errors": ["PDF parse failed"],
        "agent_runs": [],
        "cost_tracking": {},
    }


def test_paper_failure_finalize_packs_incomplete_slot_and_resets_state():
    state = _failed_state()

    result = paper_failure_finalize_node(state)

    assert result["processing_stage"] == "paper_failure_finalize"
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
    assert slot.completed is False
    assert "PDF parse failed" in slot.errors
    assert "Failed at node: ingestion" in slot.errors


def test_paper_failure_finalize_preserves_partial_data():
    state = _failed_state()
    state["metadata"] = PaperMetadata(
        title="DeepSeek-R1",
        authors=["DeepSeek-AI"],
        arxiv_id="2501.12948",
        published_date="2025-01-22",
        abstract="Reasoning model.",
        categories=["cs.AI"],
        citation_count=1,
    )
    state["method_extraction"] = MethodExtraction(
        method_name="DeepSeek-R1",
        description="Reasoning-focused LLM.",
        novelty_claim="RL-first reasoning.",
        key_components=["GRPO"],
        compared_to=["o1"],
        limitations_stated=["cost"],
    )
    state["benchmarks"] = [
        BenchmarkResult(task="MATH-500", metric="pass@1", value=97.3)
    ]

    result = paper_failure_finalize_node(state)
    slot = result["papers"][0]

    assert slot.completed is False
    assert slot.metadata is not None
    assert slot.metadata.title == "DeepSeek-R1"
    assert slot.method_extraction is not None
    assert slot.method_extraction.method_name == "DeepSeek-R1"
    assert len(slot.benchmarks) == 1
    assert slot.benchmarks[0].task == "MATH-500"


def test_paper_failure_finalize_uses_batch_url_when_present():
    state = _failed_state()
    state["batch_urls"] = [
        "https://arxiv.org/abs/2501.12948",
        "https://arxiv.org/abs/2305.14314",
    ]
    state["total_papers"] = 2
    state["current_paper_index"] = 1

    result = paper_failure_finalize_node(state)
    slot = result["papers"][0]

    assert slot.input_url == "https://arxiv.org/abs/2305.14314"
    assert slot.paper_index == 1
    assert result["current_paper_index"] == 2


def test_paper_failure_finalize_fails_on_invalid_batch_index():
    state = _failed_state()
    state["batch_urls"] = [
        "https://arxiv.org/abs/2501.12948",
        "https://arxiv.org/abs/2305.14314",
    ]
    state["total_papers"] = 2
    state["current_paper_index"] = 5

    result = paper_failure_finalize_node(state)

    assert result["processing_stage"] == "failed"
    assert result["paper_failed"] is False
    assert result["failed_node"] == "paper_failure_finalize"
