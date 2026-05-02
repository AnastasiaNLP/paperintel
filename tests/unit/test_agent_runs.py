from datetime import timezone

from models.agent_runs import AgentRun


def test_agent_run_defaults_to_running_with_timezone_aware_start():
    run = AgentRun(agent_name="report")

    assert run.id
    assert run.agent_name == "report"
    assert run.input_refs == []
    assert run.tool_calls == []
    assert run.iteration_count == 0
    assert run.status == "running"
    assert run.termination_reason is None
    assert run.finished_at is None
    assert run.started_at.tzinfo == timezone.utc


def test_agent_run_complete_updates_status_and_output_metadata():
    run = AgentRun(agent_name="report")

    result = run.complete(
        output_ref="s3://agent-output/report.json",
        confidence=0.82,
        tokens_used=123,
        cost_usd=0.01,
    )

    assert result is run
    assert run.status == "completed"
    assert run.termination_reason == "success"
    assert run.output_ref == "s3://agent-output/report.json"
    assert run.confidence == 0.82
    assert run.tokens_used == 123
    assert run.cost_usd == 0.01
    assert run.finished_at is not None


def test_agent_run_fail_records_error_termination():
    run = AgentRun(agent_name="evidence_critic")

    run.fail(output_ref="s3://agent-output/error.json")

    assert run.status == "failed"
    assert run.termination_reason == "error"
    assert run.output_ref == "s3://agent-output/error.json"
    assert run.finished_at is not None


def test_agent_run_fallback_records_fallback_status():
    run = AgentRun(agent_name="comparison_analyst")

    run.fallback(output_ref="s3://agent-output/fallback.json")

    assert run.status == "fallback_used"
    assert run.termination_reason == "fallback"
    assert run.output_ref == "s3://agent-output/fallback.json"
    assert run.finished_at is not None


def test_agent_run_model_dump_is_serializable():
    run = AgentRun(
        agent_name="searcher",
        session_id="session-1",
        job_id="job-1",
        input_refs=["query-ref"],
        model="claude-haiku",
        tool_calls=[{"tool": "search_papers", "args": {"query": "agent memory"}}],
    )
    run.complete(output_ref="output-ref")

    dumped = run.model_dump(mode="json")

    assert dumped["agent_name"] == "searcher"
    assert dumped["session_id"] == "session-1"
    assert dumped["job_id"] == "job-1"
    assert dumped["input_refs"] == ["query-ref"]
    assert dumped["tool_calls"][0]["tool"] == "search_papers"
    assert dumped["status"] == "completed"
    assert isinstance(dumped["started_at"], str)
    assert isinstance(dumped["finished_at"], str)
