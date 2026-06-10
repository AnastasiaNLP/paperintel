import logging

from services.observability import emit_event


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
