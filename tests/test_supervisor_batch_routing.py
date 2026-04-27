from agents.supervisor import (
    route_after_benchmark,
    route_after_extraction,
    route_after_finalize,
    route_after_ingestion,
    route_after_readiness,
)


def test_route_after_finalize_loops_to_ingestion_when_more_papers_remain():
    state = {
        "current_paper_index": 1,
        "total_papers": 3,
        "papers": ["slot0"],
    }
    assert route_after_finalize(state) == "ingestion"


def test_route_after_finalize_goes_to_comparator_when_batch_complete():
    state = {
        "current_paper_index": 2,
        "total_papers": 2,
        "papers": ["slot0", "slot1"],
    }
    assert route_after_finalize(state) == "comparator"


def test_route_after_finalize_ends_for_single_completed_paper():
    state = {
        "current_paper_index": 1,
        "total_papers": 1,
        "papers": ["slot0"],
    }
    assert route_after_finalize(state) == "end"


def test_route_after_ingestion_routes_paper_failure_finalize():
    assert route_after_ingestion({"processing_stage": "paper_failure_finalize"}) == (
        "paper_failure_finalize"
    )


def test_route_after_ingestion_routes_failed_to_error():
    assert route_after_ingestion({"processing_stage": "failed"}) == "error"


def test_route_after_ingestion_routes_topic_selection_to_end():
    assert route_after_ingestion({"processing_stage": "topic_selection"}) == "end"


def test_route_after_ingestion_routes_extraction_normally():
    assert route_after_ingestion({"processing_stage": "extraction"}) == "extraction"


def test_route_after_extraction_routes_paper_failure_finalize():
    assert route_after_extraction(
        {"processing_stage": "paper_failure_finalize", "needs_human_review": False}
    ) == "paper_failure_finalize"


def test_route_after_extraction_routes_failed_to_error():
    assert route_after_extraction(
        {"processing_stage": "failed", "needs_human_review": False}
    ) == "error"


def test_route_after_extraction_routes_human_review_when_needed():
    assert route_after_extraction(
        {"processing_stage": "benchmark", "needs_human_review": True}
    ) == "human_review"


def test_route_after_extraction_routes_benchmark_normally():
    assert route_after_extraction(
        {"processing_stage": "benchmark", "needs_human_review": False}
    ) == "benchmark"


def test_route_after_benchmark_routes_paper_failure_finalize():
    assert route_after_benchmark({"processing_stage": "paper_failure_finalize"}) == (
        "paper_failure_finalize"
    )


def test_route_after_benchmark_routes_failed_to_error():
    assert route_after_benchmark({"processing_stage": "failed"}) == "error"


def test_route_after_benchmark_routes_readiness_normally():
    assert route_after_benchmark({"processing_stage": "readiness"}) == "readiness"


def test_route_after_readiness_routes_paper_failure_finalize():
    assert route_after_readiness({"processing_stage": "paper_failure_finalize"}) == (
        "paper_failure_finalize"
    )


def test_route_after_readiness_routes_failed_to_error():
    assert route_after_readiness({"processing_stage": "failed"}) == "error"


def test_route_after_readiness_routes_report_normally():
    assert route_after_readiness({"processing_stage": "report"}) == "report"
