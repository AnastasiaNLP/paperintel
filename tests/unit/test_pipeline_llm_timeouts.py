from agents import benchmark, comparator, extraction, readiness


def test_extraction_llm_calls_use_policy_timeout(monkeypatch):
    captured = []

    def fake_call_text_llm(**kwargs):
        captured.append(kwargs)
        return "{}", None

    monkeypatch.setattr(extraction, "call_text_llm", fake_call_text_llm)

    assert extraction._call_llm("paper text") == ("{}", None)
    assert extraction._call_llm_repair("{bad") == ("{}", None)

    assert [call["timeout_seconds"] for call in captured] == [
        extraction._POLICY.timeout_seconds,
        extraction._POLICY.timeout_seconds,
    ]
    assert all(call["timeout_seconds"] is not None for call in captured)


def test_benchmark_llm_calls_use_policy_timeout(monkeypatch):
    captured = []

    def fake_call_text_llm(**kwargs):
        captured.append(kwargs)
        return "[]", None

    monkeypatch.setattr(benchmark, "call_text_llm", fake_call_text_llm)

    assert benchmark._call_llm(
        model="haiku",
        context_label="Benchmark Haiku LLM",
        proposed_method="method",
        tables_text="table",
        fallback_text="context",
    ) == ("[]", None)
    assert benchmark._call_llm_repair(
        "{bad",
        model="haiku",
        context_label="Benchmark repair",
    ) == ("[]", None)

    assert [call["timeout_seconds"] for call in captured] == [
        benchmark._POLICY.timeout_seconds,
        benchmark._POLICY.timeout_seconds,
    ]
    assert all(call["timeout_seconds"] is not None for call in captured)


def test_readiness_llm_calls_use_policy_timeout(monkeypatch):
    captured = []

    def fake_call_text_llm(**kwargs):
        captured.append(kwargs)
        return "{}", None

    monkeypatch.setattr(readiness, "call_text_llm", fake_call_text_llm)

    assert readiness._call_llm("{}") == ("{}", None)
    assert readiness._call_llm_repair("{bad") == ("{}", None)

    assert [call["timeout_seconds"] for call in captured] == [
        readiness._POLICY.timeout_seconds,
        readiness._POLICY.timeout_seconds,
    ]
    assert all(call["timeout_seconds"] is not None for call in captured)


def test_comparator_llm_calls_use_policy_timeout(monkeypatch):
    captured = []

    def fake_call_text_llm(**kwargs):
        captured.append(kwargs)
        return "{}", None

    monkeypatch.setattr(comparator, "call_text_llm", fake_call_text_llm)

    assert comparator._call_llm("{}") == ("{}", None)
    assert comparator._call_llm_repair("{bad") == ("{}", None)

    assert [call["timeout_seconds"] for call in captured] == [
        comparator._POLICY.timeout_seconds,
        comparator._POLICY.timeout_seconds,
    ]
    assert all(call["timeout_seconds"] is not None for call in captured)
