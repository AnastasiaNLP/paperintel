from evaluation.deterministic_metrics import (
    evaluate_benchmarks,
    evaluate_method_extraction,
    evaluate_readiness,
    evaluate_report_coverage,
    evaluate_workspace,
)
from evaluation.golden_dataset import GoldenBenchmark, load_golden_records
from models.artifacts import PaperWorkspace


def _transformer_record():
    return load_golden_records("golden_dataset/seed_5.jsonl")[0]


def _record_with_benchmark(benchmark: GoldenBenchmark):
    return _transformer_record().model_copy(update={"expected_benchmarks": [benchmark]})


def test_evaluate_method_extraction_scores_keyword_and_list_coverage():
    record = _transformer_record()

    result = evaluate_method_extraction(
        record,
        {
            "method_name": "Transformer",
            "description": (
                "Self-attention encoder-decoder architecture for sequence "
                "transduction with parallelizable training and attention-based "
                "modeling."
            ),
            "novelty_claim": (
                "A fully attention-based architecture enabling parallel "
                "computation through removal of recurrent layers and removal "
                "of convolutional layers, using attention without recurrence."
            ),
            "key_components": [
                "multi-head attention",
                "scaled dot-product attention",
                "positional encoding",
                "encoder stack",
                "decoder stack",
                "position-wise feed-forward network",
                "masked self-attention",
            ],
            "compared_to": [
                "recurrent neural networks",
                "LSTM",
                "GRU",
                "convolutional sequence models",
                "ByteNet",
                "ConvS2S",
            ],
            "limitations_stated": [],
        },
    )

    assert result.passed
    assert result.score == 1.0


def test_evaluate_benchmarks_requires_task_metric_and_value():
    record = _transformer_record()

    result = evaluate_benchmarks(
        record,
        [
            {
                "task": "machine translation",
                "metric": "BLEU",
                "value": 27.3,
                "conditions": "WMT 2014 English-to-German Transformer base",
            },
            {
                "task": "machine translation",
                "metric": "BLEU",
                "value": 28.4,
                "conditions": "WMT 2014 English-to-German Transformer big",
            },
        ],
    )

    assert not result.passed
    assert result.score == 0.5
    assert result.details["matched_count"] == 2
    assert len(result.details["missing"]) == 2


def test_evaluate_benchmarks_matches_task_and_metric_aliases():
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
        [
            {
                "task": "NQ",
                "metric": "EM",
                "value": 44.5,
                "conditions": "RAG-Seq.",
            }
        ],
    )

    assert result.passed
    assert result.score == 1.0


def test_evaluate_benchmarks_searches_conditions_in_task_and_metric():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="Natural Questions",
            metric="Exact Match",
            value=44.5,
            unit=None,
            conditions_keywords=["NQ", "EM"],
        )
    )

    result = evaluate_benchmarks(
        record,
        [
            {
                "task": "Natural Questions",
                "metric": "Exact Match",
                "value": 44.5,
                "conditions": None,
            }
        ],
    )

    assert result.passed
    assert result.score == 1.0


def test_evaluate_benchmarks_allows_single_candidate_with_missing_conditions():
    record = _record_with_benchmark(
        GoldenBenchmark(
            task="HumanEval",
            metric="pass@1",
            value=91.0,
            unit="percent",
            conditions_keywords=["Reflexion", "GPT-4", "Python"],
        )
    )

    result = evaluate_benchmarks(
        record,
        [
            {
                "task": "HumanEval",
                "metric": "pass@1",
                "value": 91.0,
                "unit": "%",
                "conditions": None,
            }
        ],
    )

    assert result.passed
    assert result.score == 1.0


def test_evaluate_benchmarks_uses_conditions_to_reject_ambiguous_candidates():
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
        [
            {
                "task": "GLUE",
                "metric": "Accuracy",
                "value": 80.5,
                "unit": "%",
                "conditions": "BERTBASE single-model",
            },
            {
                "task": "GLUE",
                "metric": "Accuracy",
                "value": 80.5,
                "unit": "%",
                "conditions": "BERTLARGE ensemble",
            },
        ],
    )

    assert not result.passed
    assert result.score == 0.0


def test_evaluate_benchmarks_does_not_match_on_value_only():
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
        [
            {
                "task": "TriviaQA",
                "metric": "Accuracy",
                "value": 44.5,
                "conditions": "RAG-Sequence",
            }
        ],
    )

    assert not result.passed
    assert result.score == 0.0


def test_evaluate_readiness_uses_framework_integrations_field():
    record = _transformer_record()

    result = evaluate_readiness(
        record,
        {
            "has_open_code": True,
            "code_url": "https://github.com/tensorflow/tensor2tensor",
            "huggingface_model": None,
            "framework_integrations": ["TensorFlow"],
            "min_gpu_requirement": None,
            "dependencies": [],
            "maturity_level": "production_ready",
        },
    )

    assert result.passed
    assert result.score == 1.0


def test_evaluate_report_coverage_scores_missing_terms():
    record = _transformer_record()

    result = evaluate_report_coverage(
        record,
        "The report mentions positional encoding and multi-head attention.",
    )

    assert not result.passed
    assert result.score == 0.4


def test_evaluate_workspace_combines_deterministic_checks():
    record = _transformer_record()
    workspace = PaperWorkspace(
        session_id="session-1",
        paper_id=record.paper_id,
        source_url=record.source_url,
        pipeline_stage="chunk_and_index",
        method_extraction_json={
            "method_name": "Transformer",
            "description": "self-attention encoder-decoder architecture",
            "novelty_claim": "attention without recurrence",
            "key_components": ["multi-head attention"],
            "compared_to": ["LSTM"],
            "limitations_stated": [],
        },
        benchmarks_json=[],
        readiness_json={
            "has_open_code": True,
            "code_url": "https://github.com/tensorflow/tensor2tensor",
            "huggingface_model": None,
            "framework_integrations": ["TensorFlow"],
            "min_gpu_requirement": None,
            "dependencies": [],
            "maturity_level": "experimental",
        },
        full_markdown_report="attention mechanism and positional encoding",
    )

    result = evaluate_workspace(record, workspace)

    assert result.paper_id == "1706.03762"
    assert not result.passed
    assert {check.name for check in result.checks} == {
        "method_extraction",
        "benchmarks",
        "readiness",
        "report_coverage",
    }
    assert 0 < result.score < 1
