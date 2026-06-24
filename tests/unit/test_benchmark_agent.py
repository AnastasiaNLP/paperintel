from agents import benchmark
from models.schemas import BenchmarkResult, BenchmarkResultV02, MethodExtraction


def _state(raw_text: str = "Table 1 reports benchmark results on Natural Questions."):
    return {
        "raw_text": raw_text,
        "pdf_path": "/tmp/paper.pdf",
        "text_by_page": {1: raw_text},
        "method_extraction": MethodExtraction(
            method_name="RAG",
            description="retrieval augmented generation",
            novelty_claim="retrieval augmented generation",
            key_components=[],
            compared_to=[],
            limitations_stated=[],
        ),
    }


def test_call_llm_formats_proposed_method_tables_and_fallback(monkeypatch):
    captured = {}

    def fake_call_text_llm(**kwargs):
        captured.update(kwargs)
        return "[]", None

    monkeypatch.setattr(benchmark, "call_text_llm", fake_call_text_llm)

    result = benchmark._call_llm(
        model="haiku",
        context_label="Benchmark Haiku LLM",
        proposed_method="RAG",
        tables_text="Table 1:\nRAG-Sequence | NQ | 44.5",
        fallback_text="[Benchmark context window 1]\nNQ uses Exact Match.",
    )

    assert result == ("[]", None)
    assert captured["requested_model"] == "haiku"
    assert "Proposed method: RAG" in captured["user_content"]
    assert "## Extracted PDF tables with page context" in captured["user_content"]
    assert "RAG-Sequence | NQ | 44.5" in captured["user_content"]
    assert "## Fallback paper text context" in captured["user_content"]
    assert "NQ uses Exact Match" in captured["user_content"]


def test_fallback_context_is_included_even_when_tables_exist(monkeypatch):
    calls = []

    monkeypatch.setattr(
        benchmark,
        "extract_tables",
        lambda path: [{"page": 1, "rows": [["Model", "NQ"], ["RAG", "44.5"]]}],
    )
    monkeypatch.setattr(benchmark.settings, "haiku_model", "haiku", raising=False)
    monkeypatch.setattr(benchmark.settings, "sonnet_model", "sonnet", raising=False)

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return "[]", None

    monkeypatch.setattr(benchmark, "_call_llm", fake_call_llm)

    result = benchmark.benchmark_analyst_agent(
        _state(
            "Table 1 benchmark results: RAG-Sequence gets 44.5 Exact Match "
            "on Natural Questions."
        )
    )

    assert result["processing_stage"] == "readiness"
    assert calls[0]["model"] == "haiku"
    assert "Natural Questions" in calls[0]["fallback_text"]
    assert "RAG-Sequence" in calls[0]["fallback_text"]


def test_sonnet_fallback_triggers_for_low_row_count_with_context(monkeypatch):
    calls = []

    monkeypatch.setattr(
        benchmark,
        "extract_tables",
        lambda path: [{"page": 1, "rows": [["Model", "NQ"], ["RAG", "44.5"]]}],
    )
    monkeypatch.setattr(benchmark.settings, "haiku_model", "haiku", raising=False)
    monkeypatch.setattr(benchmark.settings, "sonnet_model", "sonnet", raising=False)

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "haiku":
            return (
                '[{"task":"Natural Questions","metric":"Exact Match","value":44.5,'
                '"unit":null,"baseline_comparison":null,'
                '"conditions":"RAG-Sequence"}]',
                None,
            )
        return (
            '['
            '{"task":"Natural Questions","metric":"Exact Match","value":44.5,'
            '"unit":null,"baseline_comparison":null,'
            '"conditions":"RAG-Sequence"},'
            '{"task":"TriviaQA","metric":"Exact Match","value":68.0,'
            '"unit":null,"baseline_comparison":null,'
            '"conditions":"RAG-Sequence"}'
            ']',
            None,
        )

    monkeypatch.setattr(benchmark, "_call_llm", fake_call_llm)

    result = benchmark.benchmark_analyst_agent(
        _state(
            "Table 1 benchmark results: RAG-Sequence gets 44.5 Exact Match "
            "on Natural Questions and 68.0 Exact Match on TriviaQA."
        )
    )

    assert [call["model"] for call in calls] == ["haiku", "sonnet"]
    assert len(result["benchmarks"]) == 2


def test_parse_benchmarks_preserves_v01_contract_for_v01_json():
    results, error = benchmark._parse_benchmarks(
        '['
        '{"task":"English-to-German translation","metric":"BLEU",'
        '"value":28.4,"unit":null,"baseline_comparison":null,'
        '"conditions":"Transformer big"}'
        ']'
    )

    assert error is None
    assert len(results) == 1
    assert type(results[0]) is BenchmarkResult
    assert results[0].model_dump() == {
        "task": "English-to-German translation",
        "metric": "BLEU",
        "value": 28.4,
        "unit": None,
        "baseline_comparison": None,
        "conditions": "Transformer big",
    }


def test_parse_benchmarks_accepts_v02_contract_and_keeps_compat_conditions():
    results, error = benchmark._parse_benchmarks(
        '['
        '{"task":"MNLI-matched","dataset":"GLUE","metric":"Acc",'
        '"value":"91.7%","unit":"%","baseline_comparison":null,'
        '"conditions_keywords":["LoRA","GPT-3"],'
        '"source_table_or_figure":"Table 4","reported_as":"main_table",'
        '"value_type":"absolute",'
        '"evidence_anchor":{"page":8,"section":"5 EMPIRICAL EXPERIMENTS",'
        '"table_or_figure":"Table 4"},'
        '"difficulty_tags":["table_heavy","multi_dataset"]}'
        ']'
    )

    assert error is None
    assert len(results) == 1
    row = results[0]
    assert isinstance(row, BenchmarkResultV02)
    assert row.task == "MNLI-m"
    assert row.dataset == "GLUE"
    assert row.metric == "Accuracy"
    assert row.value == 91.7
    assert row.unit == "percent"
    assert row.conditions == "LoRA GPT-3"
    assert row.conditions_keywords == ["LoRA", "GPT-3"]
    assert row.source_table_or_figure == "Table 4"
    assert row.reported_as == "main_table"
    assert row.value_type == "absolute"
    assert row.evidence_anchor is not None
    assert row.evidence_anchor.page == 8
    assert row.evidence_anchor.table_or_figure == "Table 4"
    assert row.difficulty_tags == ["table_heavy", "multi_dataset"]


def test_parse_benchmarks_accepts_v02_row_with_null_dataset():
    results, error = benchmark._parse_benchmarks(
        '['
        '{"task":"Deep memory retrieval","dataset":null,"metric":"Rouge-L",'
        '"value":0.827,"unit":null,'
        '"conditions_keywords":["MemGPT","GPT-4 Turbo backbone"],'
        '"source_section":"Experiments","evidence_confidence":0.9,'
        '"higher_is_better":true}'
        ']'
    )

    assert error is None
    assert len(results) == 1
    row = results[0]
    assert isinstance(row, BenchmarkResultV02)
    assert row.task == "Deep memory retrieval"
    assert row.dataset is None
    assert row.metric == "ROUGE-L"
    assert row.conditions_keywords == ["MemGPT", "GPT-4 Turbo backbone"]
    assert row.source_section == "Experiments"
    assert row.evidence_confidence == 0.9
    assert row.higher_is_better is True


def test_parse_benchmarks_tolerates_invalid_v02_secondary_metadata():
    results, error = benchmark._parse_benchmarks(
        '['
        '{"task":"Deep memory retrieval","dataset":null,"metric":"Rouge-L",'
        '"value":0.827,"unit":null,'
        '"conditions_keywords":["MemGPT","GPT-4 Turbo backbone"],'
        '"evidence_confidence":"high",'
        '"higher_is_better":"probably",'
        '"evidence_anchor":{"page":"unknown","section":"Experiments",'
        '"table_or_figure":"Table 1"}}'
        ']'
    )

    assert error is None
    assert len(results) == 1
    row = results[0]
    assert isinstance(row, BenchmarkResultV02)
    assert row.evidence_confidence is None
    assert row.higher_is_better is None
    assert row.evidence_anchor is not None
    assert row.evidence_anchor.page is None
    assert row.evidence_anchor.section == "Experiments"


def test_parse_benchmarks_skips_invalid_rows_without_raising():
    results, error = benchmark._parse_benchmarks(
        '['
        '{"task":"valid","metric":"Accuracy","value":1.0,'
        '"dataset":"MMLU","conditions_keywords":["zero-shot"]},'
        '{"task":"invalid","metric":"Accuracy","value":"not numeric",'
        '"dataset":"MMLU","conditions_keywords":["zero-shot"]}'
        ']'
    )

    assert error is None
    assert len(results) == 1
    assert results[0].task == "valid"


def test_empty_benchmark_result_includes_context_diagnostics(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "extract_tables",
        lambda path: [
            {
                "page": 1,
                "needs_vision": True,
                "rows": [["Model", "NQ"], ["RAG", "44.5"]],
            }
        ],
    )
    monkeypatch.setattr(benchmark.settings, "haiku_model", "haiku", raising=False)
    monkeypatch.setattr(benchmark.settings, "sonnet_model", "sonnet", raising=False)
    monkeypatch.setattr(benchmark, "_call_llm", lambda **kwargs: ("[]", None))

    result = benchmark.benchmark_analyst_agent(
        _state("Table 1 benchmark results: Natural Questions Exact Match 44.5.")
    )

    messages = [error.message for error in result["errors"]]
    assert any("tables_count=1" in message for message in messages)
    assert any("complex_tables=True" in message for message in messages)
    assert any("fallback_context_chars=" in message for message in messages)
