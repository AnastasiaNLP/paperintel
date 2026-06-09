import importlib
from types import SimpleNamespace

import httpx


def _module(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "langchain-key")
    return importlib.import_module("agents.llm_provider")


def test_call_text_llm_openai_uses_timeout_seconds(monkeypatch):
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


def test_call_text_llm_openai_timeout_returns_neutral_error(monkeypatch):
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


def test_call_text_llm_anthropic_uses_timeout_seconds(monkeypatch):
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
