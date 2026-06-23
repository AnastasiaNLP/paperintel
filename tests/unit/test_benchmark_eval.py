from evaluation.benchmark_eval import (
    CONDITION_MISMATCH,
    DATASET_MISMATCH,
    MATCHED,
    MISSING,
    UNIT_MISMATCH,
    VALUE_MISMATCH,
    evaluate_benchmarks,
)
from evaluation.fixtures import build_perfect_workspace
from evaluation.golden_dataset import GoldenBenchmark, load_golden_records


def _transformer_record():
    return load_golden_records("golden_dataset/seed_5.jsonl")[0]


def _record_with_benchmark(benchmark: GoldenBenchmark):
    return _transformer_record().model_copy(update={"expected_benchmarks": [benchmark]})


def test_benchmark_eval_scores_perfect_workspace():
    record = _transformer_record()
    workspace = build_perfect_workspace(record)

    result = evaluate_benchmarks(record, workspace)

    assert result.passed
    assert result.score == 1.0
    assert result.expected_count == len(record.expected_benchmarks)
    assert result.matched_count == len(record.expected_benchmarks)
    assert result.missing_count == 0
    assert all(diagnostic.status == MATCHED for diagnostic in result.diagnostics)


def test_benchmark_eval_matches_task_metric_value_and_unit():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="GLUE",
            metric="Accuracy",
            value=80.5,
            unit="percent",
            conditions_keywords=["BERTLARGE"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "GLUE",
                    "metric": "Accuracy",
                    "value": 80.5,
                    "unit": "%",
                    "conditions": "BERTLARGE test set",
                }
            ]
        },
    )

    assert result.passed
    assert result.diagnostics[0].component_scores == {
        "task": 1.0,
        "metric": 1.0,
        "value": 1.0,
        "unit": 1.0,
        "dataset": 1.0,
        "conditions": 1.0,
    }


def test_benchmark_eval_matches_task_and_metric_aliases():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="Natural Questions",
            metric="Exact Match",
            value=44.5,
            unit=None,
            conditions_keywords=["RAG-Sequence"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "NQ",
                    "metric": "EM",
                    "value": 44.5,
                    "conditions": "RAG-Seq.",
                }
            ]
        },
    )

    assert result.passed
    assert result.diagnostics[0].status == MATCHED


def test_benchmark_eval_value_mismatch_is_diagnostic():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="GLUE",
            metric="Accuracy",
            value=80.5,
            unit="percent",
            conditions_keywords=["BERTLARGE"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "GLUE",
                    "metric": "Accuracy",
                    "value": 79.0,
                    "unit": "%",
                    "conditions": "BERTLARGE test set",
                }
            ]
        },
    )

    diagnostic = result.diagnostics[0]
    assert not result.passed
    assert diagnostic.status == VALUE_MISMATCH
    assert diagnostic.near_matches[0]["component_scores"]["value"] == 0.0


def test_benchmark_eval_unit_mismatch_is_diagnostic():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="Latency",
            metric="Runtime",
            value=10.0,
            unit="ms",
            conditions_keywords=["GPU"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "Latency",
                    "metric": "Runtime",
                    "value": 10.0,
                    "unit": "seconds",
                    "conditions": "GPU",
                }
            ]
        },
    )

    diagnostic = result.diagnostics[0]
    assert diagnostic.status == UNIT_MISMATCH
    assert diagnostic.near_matches[0]["component_scores"]["unit"] == 0.0


def test_benchmark_eval_condition_mismatch_is_diagnostic():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="GLUE",
            metric="Accuracy",
            value=80.5,
            unit="percent",
            conditions_keywords=["BERTLARGE", "single-model"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "GLUE",
                    "metric": "Accuracy",
                    "value": 80.5,
                    "unit": "%",
                    "conditions": "BERTBASE ensemble",
                }
            ]
        },
    )

    diagnostic = result.diagnostics[0]
    assert diagnostic.status == CONDITION_MISMATCH
    assert diagnostic.component_scores["conditions"] == 0.0


def test_benchmark_eval_empty_v02_conditions_keywords_can_match():
    records = load_golden_records("golden_dataset/paperintel_60_v2_2.jsonl")
    record = next(
        record
        for record in records
        if any(not row.conditions_keywords for row in record.expected_benchmarks)
    )
    expected = next(row for row in record.expected_benchmarks if not row.conditions_keywords)

    result = evaluate_benchmarks(
        record.model_copy(update={"expected_benchmarks": [expected]}),
        {
            "benchmarks_json": [
                {
                    "task": expected.task,
                    "metric": expected.metric,
                    "value": expected.value,
                    "unit": expected.unit,
                    "dataset": expected.dataset,
                    "conditions": expected.conditions,
                }
            ]
        },
    )

    assert result.passed
    assert result.diagnostics[0].component_scores["conditions"] == 1.0


def test_benchmark_eval_v02_wrong_dataset_does_not_match_empty_conditions_keywords():
    records = load_golden_records("golden_dataset/paperintel_60_v2_2.jsonl")
    record = next(
        record
        for record in records
        if any(row.dataset and not row.conditions_keywords for row in record.expected_benchmarks)
    )
    expected = next(
        row
        for row in record.expected_benchmarks
        if row.dataset and not row.conditions_keywords
    )

    result = evaluate_benchmarks(
        record.model_copy(update={"expected_benchmarks": [expected]}),
        {
            "benchmarks_json": [
                {
                    "task": expected.task,
                    "metric": expected.metric,
                    "value": expected.value,
                    "unit": expected.unit,
                    "dataset": "definitely-wrong-dataset",
                    "conditions": expected.conditions,
                }
            ]
        },
    )

    diagnostic = result.diagnostics[0]
    assert not result.passed
    assert diagnostic.status == DATASET_MISMATCH
    assert diagnostic.component_scores["dataset"] == 0.0


def test_benchmark_eval_v02_diagnostic_includes_metadata():
    record = load_golden_records("golden_dataset/paperintel_60_v2_2.jsonl")[0]
    expected = record.expected_benchmarks[0]

    result = evaluate_benchmarks(
        record.model_copy(update={"expected_benchmarks": [expected]}),
        {
            "benchmarks_json": [
                {
                    "task": expected.task,
                    "metric": expected.metric,
                    "value": expected.value,
                    "unit": expected.unit,
                    "dataset": expected.dataset,
                    "conditions": expected.conditions,
                }
            ]
        },
    )

    expected_row = result.diagnostics[0].expected
    assert expected_row.dataset == expected.dataset
    assert expected_row.source_section == expected.source_section
    assert expected_row.source_table_or_figure == expected.source_table_or_figure
    assert expected_row.reported_as == expected.reported_as
    assert expected_row.value_type == expected.value_type
    assert expected_row.evidence_anchor == expected.evidence_anchor.model_dump()
    assert expected_row.evidence_confidence == expected.evidence_confidence
    assert expected_row.paper_family == record.paper_family
    assert expected_row.difficulty_tags == record.difficulty_tags
    assert expected_row.label_quality == record.label_quality


def test_benchmark_eval_accepts_raw_dict_workspace():
    record = _transformer_record()
    workspace = build_perfect_workspace(record).model_dump()

    result = evaluate_benchmarks(record, workspace)

    assert result.passed
    assert result.score == 1.0


def test_benchmark_eval_missing_row_is_diagnostic():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="GLUE",
            metric="Accuracy",
            value=80.5,
            unit="percent",
            conditions_keywords=["BERTLARGE"],
        )
    )

    result = evaluate_benchmarks(record, {"benchmarks_json": []})

    diagnostic = result.diagnostics[0]
    assert not result.passed
    assert result.missing_count == 1
    assert diagnostic.status == MISSING
    assert diagnostic.near_matches == []


def test_benchmark_eval_failure_status_prefers_task_metric_candidate_hierarchy():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="GLUE",
            metric="Accuracy",
            value=80.5,
            unit="percent",
            conditions_keywords=["BERTLARGE", "single-model"],
        )
    )

    result = evaluate_benchmarks(
        record,
        {
            "benchmarks_json": [
                {
                    "task": "Wrong Task",
                    "metric": "Accuracy",
                    "value": 80.5,
                    "unit": "%",
                    "conditions": "BERTLARGE single-model",
                },
                {
                    "task": "GLUE",
                    "metric": "Accuracy",
                    "value": 80.5,
                    "unit": "%",
                    "conditions": "BERTBASE ensemble",
                },
            ]
        },
    )

    diagnostic = result.diagnostics[0]
    assert diagnostic.status == CONDITION_MISMATCH
    assert diagnostic.component_scores == {
        "task": 1.0,
        "metric": 1.0,
        "value": 1.0,
        "unit": 1.0,
        "dataset": 1.0,
        "conditions": 0.0,
    }
