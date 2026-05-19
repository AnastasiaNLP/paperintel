from __future__ import annotations

from datetime import datetime, timezone

from evaluation.golden_dataset import GoldenDatasetRecord
from models.artifacts import PaperWorkspace


FIXTURE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_perfect_workspace(record: GoldenDatasetRecord) -> PaperWorkspace:
    method = record.expected_method_extraction
    readiness = record.expected_readiness
    return PaperWorkspace(
        id=f"fixture-workspace-{record.paper_id}",
        session_id="evaluation-fixture-session",
        paper_id=record.paper_id,
        title=record.title,
        source_url=record.source_url,
        pipeline_stage="chunk_and_index",
        method_extraction_json={
            "method_name": method.method_name,
            "description": " ".join(method.description_keywords),
            "novelty_claim": " ".join(method.novelty_keywords),
            "key_components": method.key_components,
            "compared_to": method.compared_to,
            "limitations_stated": method.limitations_stated,
        },
        benchmarks_json=[
            {
                "task": benchmark.task,
                "metric": benchmark.metric,
                "value": benchmark.value,
                "unit": benchmark.unit,
                "conditions": " ".join(benchmark.conditions_keywords),
            }
            for benchmark in record.expected_benchmarks
        ],
        readiness_json={
            "has_open_code": readiness.has_open_code,
            "code_url": readiness.code_url,
            "huggingface_model": readiness.huggingface_model,
            "framework_integrations": readiness.expected_framework_integrations,
            "min_gpu_requirement": readiness.min_gpu_requirement,
            "dependencies": readiness.dependencies,
            "maturity_level": readiness.maturity_level,
        },
        full_markdown_report=" ".join(record.expected_report_coverage.must_mention),
        created_at=FIXTURE_TIMESTAMP,
        updated_at=FIXTURE_TIMESTAMP,
    )


def build_partial_workspace(record: GoldenDatasetRecord) -> PaperWorkspace:
    workspace = build_perfect_workspace(record)
    return workspace.model_copy(
        update={
            "benchmarks_json": workspace.benchmarks_json[:1],
            "full_markdown_report": " ".join(
                record.expected_report_coverage.must_mention[:2]
            ),
        }
    )
