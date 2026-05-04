from agents.error_utils import fatal_error, paper_error
from agents.paper_failure_finalize import paper_failure_finalize_node
from agents.report_finalize import report_finalize_node
from models.errors import (
    ErrorCodes,
    StructuredError,
    error_message,
    error_messages,
    make_error,
    normalize_error,
)
from models.schemas import EngineerReport


def test_make_error_creates_structured_error():
    error = make_error(
        ErrorCodes.PAPER_ERROR,
        "PDF parse failed",
        node="ingestion",
        severity="error",
        recoverable=True,
        reason="bad_pdf",
    )

    assert isinstance(error, StructuredError)
    assert error.code == ErrorCodes.PAPER_ERROR
    assert error.message == "PDF parse failed"
    assert error.node == "ingestion"
    assert error.severity == "error"
    assert error.recoverable is True
    assert error.details == {"reason": "bad_pdf"}


def test_normalize_error_converts_string_to_warning():
    error = normalize_error("Benchmark warning", default_node="benchmark")

    assert isinstance(error, StructuredError)
    assert error.code == ErrorCodes.WARNING
    assert error.message == "Benchmark warning"
    assert error.node == "benchmark"
    assert error.severity == "warning"
    assert error.recoverable is True


def test_error_message_handles_structured_and_string_errors():
    structured = make_error(ErrorCodes.FATAL_ERROR, "fatal", recoverable=False)

    assert error_message(structured) == "fatal"
    assert error_message("plain") == "plain"
    assert error_messages([structured, "plain"]) == ["fatal", "plain"]


def test_paper_error_returns_structured_error_and_batch_finalize_stage():
    result = paper_error(
        {
            "batch_urls": [
                "https://arxiv.org/abs/2501.12948",
                "https://arxiv.org/abs/2305.14314",
            ],
            "total_papers": 2,
        },
        "PDF parse failed",
        "ingestion",
    )

    assert result["processing_stage"] == "paper_failure_finalize"
    assert result["paper_failed"] is True
    assert result["paper_failure_reason"] == "PDF parse failed"
    assert result["failed_node"] == "ingestion"
    assert isinstance(result["errors"][0], StructuredError)
    assert result["errors"][0].recoverable is True


def test_fatal_error_returns_nonrecoverable_structured_error():
    result = fatal_error("Invalid state", "supervisor")

    assert result["processing_stage"] == "failed"
    assert result["paper_failed"] is False
    assert result["failed_node"] == "supervisor"
    assert isinstance(result["errors"][0], StructuredError)
    assert result["errors"][0].code == ErrorCodes.FATAL_ERROR
    assert result["errors"][0].severity == "fatal"
    assert result["errors"][0].recoverable is False


def test_paper_failure_finalize_converts_structured_errors_to_slot_messages():
    state = {
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "papers": [],
        "metadata": None,
        "raw_text": "paper text",
        "pdf_path": None,
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
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
        "errors": [make_error(ErrorCodes.PAPER_ERROR, "PDF parse failed", node="ingestion")],
        "agent_runs": [],
        "messages": [],
        "cost_tracking": {},
    }

    result = paper_failure_finalize_node(state)
    slot = result["papers"][0]

    assert slot.errors == ["PDF parse failed", "Failed at node: ingestion"]


def test_report_finalize_preserves_string_compatible_slot_errors():
    state = {
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "papers": [],
        "metadata": None,
        "raw_text": "paper text",
        "pdf_path": None,
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
        "engineer_report": EngineerReport(
            executive_summary="Summary",
            key_innovation="Innovation",
            practical_implications="Implications",
            implementation_difficulty="moderate",
            recommended_action="prototype",
            action_reasoning="Reasoning",
        ),
        "full_markdown_report": "# Report",
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "completed",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
        "errors": [
            make_error(ErrorCodes.WARNING, "Structured warning", severity="warning"),
            "Plain warning",
        ],
        "agent_runs": [],
        "messages": [],
        "cost_tracking": {},
    }

    result = report_finalize_node(state)
    slot = result["papers"][0]

    assert slot.errors == ["Structured warning", "Plain warning"]
