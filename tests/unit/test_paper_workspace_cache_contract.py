from storage.models import PaperWorkspaceORM
from storage.repositories import _workspace_orm_ready_for_cache


def _workspace(**overrides):
    values = {
        "pipeline_stage": "completed",
        "full_markdown_report": "# Report",
        "finalized_report_json": {"recommended_action": "prototype"},
        "method_extraction_json": {"method_name": "Method"},
        "benchmarks_json": [
            {"task": "translation", "metric": "BLEU", "value": 28.4}
        ],
        "benchmark_candidates_json": [
            {
                "task": "translation",
                "metric": "BLEU",
                "value": 28.4,
                "selection_status": "accepted",
            }
        ],
        "benchmark_extractor_version": "legacy_mirror_v1",
        "readiness_json": {"maturity_level": "experimental"},
    }
    values.update(overrides)
    return PaperWorkspaceORM(**values)


def test_workspace_cache_requires_benchmark_candidate_contract():
    workspace = _workspace(
        benchmark_candidates_json=[],
        benchmark_extractor_version=None,
    )

    assert not _workspace_orm_ready_for_cache(workspace)


def test_workspace_cache_accepts_ready_workspace_with_candidate_contract():
    workspace = _workspace()

    assert _workspace_orm_ready_for_cache(workspace)


def test_workspace_cache_allows_empty_candidates_only_when_no_benchmarks_exist():
    workspace = _workspace(
        benchmarks_json=[],
        benchmark_candidates_json=[],
        benchmark_extractor_version="legacy_mirror_v1",
    )

    assert _workspace_orm_ready_for_cache(workspace)
