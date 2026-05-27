import pytest
from pydantic import ValidationError

from models.jobs import WorkflowJob


def test_workflow_job_defaults_to_queued_with_single_attempt_budget():
    job = WorkflowJob(
        session_id="session-1",
        kind="analyze_paper",
        input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
    )

    assert job.id
    assert job.status == "queued"
    assert job.attempts == 0
    assert job.max_attempts == 1
    assert job.result_json is None
    assert job.error_json is None
    assert job.created_at.tzinfo is not None
    assert job.updated_at.tzinfo is not None


def test_workflow_job_rejects_invalid_kind_and_status():
    with pytest.raises(ValidationError):
        WorkflowJob(session_id="session-1", kind="invalid", input_json={})

    with pytest.raises(ValidationError):
        WorkflowJob(
            session_id="session-1",
            kind="compare",
            status="done",
            input_json={},
        )


def test_workflow_job_rejects_invalid_attempt_counts():
    with pytest.raises(ValidationError):
        WorkflowJob(
            session_id="session-1",
            kind="compare",
            input_json={},
            attempts=-1,
        )

    with pytest.raises(ValidationError):
        WorkflowJob(
            session_id="session-1",
            kind="compare",
            input_json={},
            max_attempts=0,
        )
