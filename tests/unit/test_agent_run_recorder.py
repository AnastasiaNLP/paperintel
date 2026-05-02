import pytest

from agents.agent_run_recorder import InMemoryAgentRunRecorder


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
