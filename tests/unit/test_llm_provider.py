import importlib
import logging
from types import SimpleNamespace

import httpx


def _module(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "langchain-key")
    return importlib.import_module("agents.llm_provider")


def test_llm_error_termination_reason_normalizes_timeout(monkeypatch):
    llm_provider = _module(monkeypatch)

    assert (
        llm_provider.llm_error_termination_reason("Answer Agent call timed out")
        == "timeout"
    )
    assert (
        llm_provider.llm_error_termination_reason(
            "Answer Agent failed",
            default="fallback",
        )
        == "fallback"
    )


def test_call_text_llm_openai_uses_timeout_seconds(monkeypatch, caplog):
    llm_provider = _module(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(
        llm_provider,
        "settings",
        SimpleNamespace(
            llm_provider="openai",
            openai_model="gpt-test",
            openai_api_key="openai-key",
        ),
    )
    monkeypatch.setattr(llm_provider.httpx, "post", fake_post)

    with caplog.at_level(logging.INFO, logger="agents.llm_provider"):
        raw, error = llm_provider.call_text_llm(
            requested_model=None,
            system_prompt="system",
            user_content="user",
            max_tokens=10,
            context_label="Test LLM",
            timeout_seconds=7.5,
        )

    assert raw == "ok"
    assert error is None
    assert calls[0]["timeout"] == 7.5
    messages = [record.getMessage() for record in caplog.records]
    event = next(
        message for message in messages if "event=llm.call.completed" in message
    )
    assert 'provider="openai"' in event
    assert 'model="gpt-test"' in event
    assert "result_size=2" in event
    assert "duration_ms=" in event


def test_call_text_llm_openai_timeout_returns_neutral_error(monkeypatch, caplog):
    llm_provider = _module(monkeypatch)

    def fake_post(url, **kwargs):
        raise httpx.TimeoutException("raw provider timeout details")

    monkeypatch.setattr(
        llm_provider,
        "settings",
        SimpleNamespace(
            llm_provider="openai",
            openai_model="gpt-test",
            openai_api_key="openai-key",
        ),
    )
    monkeypatch.setattr(llm_provider.httpx, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="agents.llm_provider"):
        raw, error = llm_provider.call_text_llm(
            requested_model=None,
            system_prompt="do not leak this system prompt",
            user_content="do not leak this user prompt",
            max_tokens=10,
            context_label="Test LLM",
            timeout_seconds=7.5,
        )

    assert raw is None
    assert error == "Test LLM call timed out"
    assert "provider timeout details" not in error
    assert "do not leak" not in error
    messages = [record.getMessage() for record in caplog.records]
    assert any("event=llm.call.timeout" in message for message in messages)
    assert any('provider="openai"' in message for message in messages)
    assert any("duration_ms=" in message for message in messages)
    assert all("do not leak" not in message for message in messages)
    assert all("provider timeout details" not in message for message in messages)


def test_call_text_llm_openai_http_error_emits_failed_duration(monkeypatch, caplog):
    llm_provider = _module(monkeypatch)

    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            500,
            request=request,
            text="raw provider body should not be in structured event",
        )
        raise httpx.HTTPStatusError(
            "raw provider failure",
            request=request,
            response=response,
        )

    monkeypatch.setattr(
        llm_provider,
        "settings",
        SimpleNamespace(
            llm_provider="openai",
            openai_model="gpt-test",
            openai_api_key="openai-key",
        ),
    )
    monkeypatch.setattr(llm_provider.httpx, "post", fake_post)

    with caplog.at_level(logging.ERROR, logger="agents.llm_provider"):
        raw, error = llm_provider.call_text_llm(
            requested_model=None,
            system_prompt="system",
            user_content="user",
            max_tokens=10,
            context_label="Test LLM",
            timeout_seconds=7.5,
        )

    assert raw is None
    assert error is not None
    messages = [record.getMessage() for record in caplog.records]
    event = next(
        message for message in messages if "event=llm.call.failed" in message
    )
    assert 'provider="openai"' in event
    assert 'model="gpt-test"' in event
    assert "duration_ms=" in event
    assert "raw provider body" not in event
    assert "raw provider failure" not in event


def test_call_text_llm_anthropic_uses_timeout_seconds(monkeypatch, caplog):
    llm_provider = _module(monkeypatch)

    class MessageClient:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text="anthropic ok")]
            )

    class AnthropicClient:
        def __init__(self):
            self.messages = MessageClient()
            self.timeouts = []

        def with_options(self, *, timeout):
            self.timeouts.append(timeout)
            return self

    client = AnthropicClient()
    monkeypatch.setattr(
        llm_provider,
        "settings",
        SimpleNamespace(llm_provider="anthropic", haiku_model="claude-test"),
    )
    monkeypatch.setattr(llm_provider, "_anthropic", lambda: client)

    with caplog.at_level(logging.INFO, logger="agents.llm_provider"):
        raw, error = llm_provider.call_text_llm(
            requested_model=None,
            system_prompt="system",
            user_content="user",
            max_tokens=10,
            context_label="Test LLM",
            timeout_seconds=3.0,
        )

    assert raw == "anthropic ok"
    assert error is None
    assert client.timeouts == [3.0]
    assert client.messages.calls[0]["model"] == "claude-test"
    messages = [record.getMessage() for record in caplog.records]
    event = next(
        message for message in messages if "event=llm.call.completed" in message
    )
    assert 'provider="anthropic"' in event
    assert 'model="claude-test"' in event
    assert "result_size=12" in event
    assert "duration_ms=" in event


def test_call_text_llm_anthropic_error_emits_failed_duration(monkeypatch, caplog):
    llm_provider = _module(monkeypatch)

    class MessageClient:
        def create(self, **kwargs):
            raise RuntimeError("raw anthropic failure")

    class AnthropicClient:
        messages = MessageClient()

        def with_options(self, *, timeout):
            return self

    monkeypatch.setattr(
        llm_provider,
        "settings",
        SimpleNamespace(llm_provider="anthropic", haiku_model="claude-test"),
    )
    monkeypatch.setattr(llm_provider, "_anthropic", lambda: AnthropicClient())

    with caplog.at_level(logging.ERROR, logger="agents.llm_provider"):
        raw, error = llm_provider.call_text_llm(
            requested_model=None,
            system_prompt="system",
            user_content="user",
            max_tokens=10,
            context_label="Test LLM",
            timeout_seconds=3.0,
        )

    assert raw is None
    assert error == "Test LLM call failed: raw anthropic failure"
    messages = [record.getMessage() for record in caplog.records]
    event = next(
        message for message in messages if "event=llm.call.failed" in message
    )
    assert 'provider="anthropic"' in event
    assert 'model="claude-test"' in event
    assert "duration_ms=" in event
    assert "raw anthropic failure" not in event
