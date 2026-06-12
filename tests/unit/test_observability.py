import logging

from services.observability import (
    attach_observability_details,
    current_trace_id,
    emit_event,
    render_prometheus_metrics,
    reset_observability_metrics,
)


def setup_function():
    reset_observability_metrics()


def test_emit_event_logs_name_and_safe_fields(caplog):
    logger = logging.getLogger("tests.observability")

    with caplog.at_level(logging.INFO, logger="tests.observability"):
        emit_event(
            logger,
            "workflow.job.failed",
            job_id="job-1",
            session_id="session-1",
            failure_class="provider_unavailable",
            retryable=True,
            retry_after_seconds=30.0,
            result_count=3,
            duration_ms=12,
        )

    message = caplog.records[-1].getMessage()
    assert "event=workflow.job.failed" in message
    assert 'job_id="job-1"' in message
    assert 'session_id="session-1"' in message
    assert 'failure_class="provider_unavailable"' in message
    assert "retryable=true" in message
    assert "retry_after_seconds=30.0" in message
    assert "result_count=3" in message
    assert "duration_ms=12" in message


def test_emit_event_drops_forbidden_and_unknown_fields(caplog):
    logger = logging.getLogger("tests.observability")

    with caplog.at_level(logging.INFO, logger="tests.observability"):
        emit_event(
            logger,
            "llm.call.failed",
            provider="openai",
            model="gpt-test",
            system_prompt="do not log",
            user_content="do not log either",
            api_key="secret-key",
            unexpected_payload="drop me",
        )

    message = caplog.records[-1].getMessage()
    assert "event=llm.call.failed" in message
    assert 'provider="openai"' in message
    assert 'model="gpt-test"' in message
    assert "do not log" not in message
    assert "secret-key" not in message
    assert "unexpected_payload" not in message


def test_emit_event_records_prometheus_counter_and_duration_histogram():
    logger = logging.getLogger("tests.observability")

    emit_event(
        logger,
        "workflow.job.failed",
        job_id="job-1",
        session_id="session-1",
        kind="analyze_paper",
        provider="anthropic",
        failure_class="provider_timeout",
        retryable=True,
        duration_ms=120,
        system_prompt="do not export",
    )

    output = render_prometheus_metrics()

    assert "# TYPE paperintel_events_total counter" in output
    assert (
        'paperintel_events_total{event="workflow.job.failed",'
        'failure_class="provider_timeout",kind="analyze_paper",'
        'provider="anthropic"} 1'
    ) in output
    assert 'paperintel_event_duration_ms_bucket{event="workflow.job.failed"' in output
    assert 'le="+Inf"} 1' in output
    assert "job-1" not in output
    assert "session-1" not in output
    assert "do not export" not in output


def test_emit_event_adds_safe_trace_id_from_environment(caplog, monkeypatch):
    logger = logging.getLogger("tests.observability")
    monkeypatch.setenv("PAPERINTEL_TRACE_ID", "trace-123")

    with caplog.at_level(logging.INFO, logger="tests.observability"):
        emit_event(logger, "agent.completed", agent_name="answer_agent")

    message = caplog.records[-1].getMessage()
    assert 'trace_id="trace-123"' in message
    assert current_trace_id() == "trace-123"

    metrics = render_prometheus_metrics()
    assert "trace-123" not in metrics


def test_attach_observability_details_preserves_existing_trace_id(monkeypatch):
    monkeypatch.setenv("PAPERINTEL_TRACE_ID", "external-trace")
    details = {"observability": {"trace_id": "existing-trace"}}

    attach_observability_details(details, duration_ms=42)

    assert details["observability"] == {
        "trace_id": "existing-trace",
        "duration_ms": 42,
    }


def test_emit_event_normalizes_supplied_trace_id(caplog):
    logger = logging.getLogger("tests.observability")

    with caplog.at_level(logging.INFO, logger="tests.observability"):
        emit_event(
            logger,
            "agent.completed",
            agent_name="answer_agent",
            trace_id=" trace-from-caller ",
        )

    assert 'trace_id="trace-from-caller"' in caplog.records[-1].getMessage()


def test_emit_event_drops_forbidden_supplied_trace_id(caplog, monkeypatch):
    logger = logging.getLogger("tests.observability")
    monkeypatch.delenv("PAPERINTEL_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_RUN_ID", raising=False)

    with caplog.at_level(logging.INFO, logger="tests.observability"):
        emit_event(
            logger,
            "agent.completed",
            agent_name="answer_agent",
            trace_id="secret-token-value",
        )

    message = caplog.records[-1].getMessage()
    assert "trace_id=" not in message
    assert "secret-token-value" not in message


def test_attach_observability_details_drops_forbidden_existing_trace_id(monkeypatch):
    monkeypatch.delenv("PAPERINTEL_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACE_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_RUN_ID", raising=False)
    details = {"observability": {"trace_id": "secret-token-value"}}

    attach_observability_details(details, duration_ms=42)

    assert details["observability"] == {"duration_ms": 42}
