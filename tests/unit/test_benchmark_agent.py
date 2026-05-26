from agents import benchmark
from models.schemas import MethodExtraction


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
