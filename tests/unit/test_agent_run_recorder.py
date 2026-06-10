import logging

import pytest

from agents.agent_run_recorder import (
    InMemoryAgentRunPersistence,
    InMemoryAgentRunRecorder,
    NoopAgentRunPersistence,
)


def test_recorder_starts_and_stores_run():
    recorder = InMemoryAgentRunRecorder()

    run = recorder.start(
        agent_name="report",
        session_id="session-1",
        job_id="job-1",
        input_refs=["input-ref"],
        model="claude-haiku",
        iteration_count=1,
    )

    stored = recorder.get(run.id)
    assert stored is run
    assert stored.agent_name == "report"
    assert stored.session_id == "session-1"
    assert stored.job_id == "job-1"
    assert stored.input_refs == ["input-ref"]
    assert stored.model == "claude-haiku"
    assert stored.iteration_count == 1
    assert stored.status == "running"


def test_recorder_complete_updates_run():
    recorder = InMemoryAgentRunRecorder()
    run = recorder.start(agent_name="report")

    completed = recorder.complete(
        run.id,
        output_ref="output-ref",
        confidence=0.9,
        tokens_used=100,
        cost_usd=0.02,
    )

    assert completed is run
    assert completed.status == "completed"
    assert completed.output_ref == "output-ref"
    assert completed.confidence == 0.9
    assert completed.tokens_used == 100
    assert completed.cost_usd == 0.02
    assert completed.finished_at is not None


def test_recorder_fail_updates_run():
    recorder = InMemoryAgentRunRecorder()
    run = recorder.start(agent_name="evidence_critic")

    failed = recorder.fail(run.id, output_ref="error-ref")

    assert failed.status == "failed"
    assert failed.termination_reason == "error"
    assert failed.output_ref == "error-ref"


def test_recorder_fallback_updates_run():
    recorder = InMemoryAgentRunRecorder()
    run = recorder.start(agent_name="comparison")

    fallback = recorder.fallback(run.id, output_ref="fallback-ref")

    assert fallback.status == "fallback_used"
    assert fallback.termination_reason == "fallback"
    assert fallback.output_ref == "fallback-ref"


def test_recorder_list_runs_preserves_started_runs():
    recorder = InMemoryAgentRunRecorder()
    first = recorder.start(agent_name="first")
    second = recorder.start(agent_name="second")

    assert recorder.list_runs() == [first, second]


def test_recorder_raises_for_missing_run():
    recorder = InMemoryAgentRunRecorder()

    with pytest.raises(KeyError, match="AgentRun not found"):
        recorder.get("missing")


def test_noop_persistence_accepts_run_without_storing():
    recorder = InMemoryAgentRunRecorder()
    persistence = NoopAgentRunPersistence()
    run = recorder.start(agent_name="report")

    assert persistence.save(run) is None


def test_in_memory_persistence_records_saved_runs():
    recorder = InMemoryAgentRunRecorder()
    persistence = InMemoryAgentRunPersistence()
    first = recorder.start(agent_name="report")
    second = recorder.start(agent_name="evidence_critic")

    persistence.save(first)
    persistence.save(second)

    assert persistence.list_runs() == [first, second]


def test_in_memory_persistence_emits_started_for_running_run(caplog):
    recorder = InMemoryAgentRunRecorder()
    persistence = InMemoryAgentRunPersistence()
    run = recorder.start(
        agent_name="answer_agent",
        session_id="session-1",
        input_refs=["raw prompt text should not be logged"],
        model="claude-test",
    )

    with caplog.at_level(logging.INFO, logger="agents.agent_run_recorder"):
        persistence.save(run)

    message = caplog.records[-1].getMessage()
    assert "event=agent.started" in message
    assert f'agent_run_id="{run.id}"' in message
    assert 'agent_name="answer_agent"' in message
    assert 'session_id="session-1"' in message
    assert 'model="claude-test"' in message
    assert "raw prompt text" not in message


def test_in_memory_persistence_emits_safe_agent_completed_events(caplog):
    recorder = InMemoryAgentRunRecorder()
    persistence = InMemoryAgentRunPersistence()
    run = recorder.start(
        agent_name="answer_agent",
        session_id="session-1",
        job_id="job-1",
        input_refs=["raw prompt text should not be logged"],
        model="claude-test",
    )
    recorder.complete(
        run.id,
        output_ref="raw output text should not be logged",
        termination_reason="success",
    )

    with caplog.at_level(logging.INFO, logger="agents.agent_run_recorder"):
        persistence.save(run)

    messages = [record.getMessage() for record in caplog.records]
    completed = next(
        message for message in messages if "event=agent.completed" in message
    )
    assert all("event=agent.started" not in message for message in messages)
    assert f'agent_run_id="{run.id}"' in completed
    assert 'status="completed"' in completed
    assert 'termination_reason="success"' in completed
    assert "duration_ms=" in completed
    assert all("raw prompt text" not in message for message in messages)
    assert all("raw output text" not in message for message in messages)


def test_in_memory_persistence_deduplicates_terminal_events(caplog):
    recorder = InMemoryAgentRunRecorder()
    persistence = InMemoryAgentRunPersistence()
    run = recorder.start(agent_name="answer_agent", session_id="session-1")
    recorder.complete(run.id)

    with caplog.at_level(logging.INFO, logger="agents.agent_run_recorder"):
        persistence.save(run)
        run.details["second_save"] = True
        persistence.save(run)

    messages = [record.getMessage() for record in caplog.records]
    assert sum("event=agent.completed" in message for message in messages) == 1


def test_in_memory_persistence_emits_safe_agent_failed_event(caplog):
    recorder = InMemoryAgentRunRecorder()
    persistence = InMemoryAgentRunPersistence()
    run = recorder.start(agent_name="citation_critic", session_id="session-1")
    recorder.fail(run.id, termination_reason="timeout")

    with caplog.at_level(logging.INFO, logger="agents.agent_run_recorder"):
        persistence.save(run)

    messages = [record.getMessage() for record in caplog.records]
    failed = next(message for message in messages if "event=agent.failed" in message)
    assert f'agent_run_id="{run.id}"' in failed
    assert 'agent_name="citation_critic"' in failed
    assert 'status="failed"' in failed
    assert 'termination_reason="timeout"' in failed
    assert 'failure_class="timeout"' in failed
    assert "duration_ms=" in failed
