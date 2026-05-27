from dataclasses import dataclass, field

import pytest

from models.jobs import WorkflowJob
from models.session import HandlerResult
from workers.workflow_worker import (
    UnsupportedWorkflowJobKindError,
    WorkflowJobExecutionError,
    WorkflowJobExecutor,
    WorkflowWorker,
    serialize_exception,
    serialize_handler_result,
)


class FakeService:
    def __init__(self) -> None:
        self.analyze_calls = []
        self.analyze_selected_calls = []
        self.fail_analyze = False

    def analyze_paper(self, session_id, paper_url):
        self.analyze_calls.append((session_id, paper_url))
        if self.fail_analyze:
            raise RuntimeError("analysis down")
        return HandlerResult(
            session_id=session_id,
            response_text="analysis complete",
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=["1706.03762"],
            artifact_refs=["paper_workspace:1706.03762"],
            comparison_markdown="# Comparison",
            needs_analysis=False,
            needs_discovery=False,
            selected_candidate_ids=["candidate-1"],
            search_warnings=["s2 unavailable"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_selected_papers(self, session_id):
        self.analyze_selected_calls.append(session_id)
        return HandlerResult(
            session_id=session_id,
            response_text="selected analysis complete",
            phase="qa",
            intent="analyze_paper",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )


@dataclass
class FakeRepository:
    jobs: list[WorkflowJob] = field(default_factory=list)
    succeeded: list[tuple[str, dict]] = field(default_factory=list)
    failed: list[tuple[str, dict]] = field(default_factory=list)
    claim_calls: list[dict] = field(default_factory=list)
    mark_running_calls: list[tuple[str, str]] = field(default_factory=list)

    def claim_next(self, *, worker_id, kinds=None):
        self.claim_calls.append({"worker_id": worker_id, "kinds": kinds})
        for index, job in enumerate(self.jobs):
            if kinds is None or job.kind in kinds:
                claimed = self.jobs.pop(index)
                return claimed.model_copy(
                    update={
                        "status": "running",
                        "locked_by": worker_id,
                        "attempts": claimed.attempts + 1,
                    }
                )
        return None

    def mark_succeeded(self, job_id, *, result_json):
        self.succeeded.append((job_id, result_json))
        return WorkflowJob(
            id=job_id,
            session_id="session-1",
            kind="analyze_paper",
            status="succeeded",
            input_json={},
            result_json=result_json,
            attempts=1,
        )

    def mark_failed(self, job_id, *, error_json):
        self.failed.append((job_id, error_json))
        return WorkflowJob(
            id=job_id,
            session_id="session-1",
            kind="analyze_paper",
            status="failed",
            input_json={},
            error_json=error_json,
            attempts=1,
        )

    def mark_running(self, job_id, *, worker_id):
        self.mark_running_calls.append((job_id, worker_id))
        raise AssertionError("worker should start jobs with claim_next only")


def _job(kind="analyze_paper", input_json=None):
    if input_json is None:
        input_json = {"paper_url": "https://arxiv.org/abs/1706.03762"}
    return WorkflowJob(
        session_id="session-1",
        kind=kind,
        input_json=input_json,
    )


def test_serialize_handler_result_is_transport_safe():
    result = HandlerResult(
        session_id="session-1",
        response_text="done",
        phase="qa",
        intent="analyze_paper",
        referenced_paper_ids=["paper-1"],
        artifact_refs=["paper_workspace:paper-1"],
        selected_candidate_ids=["candidate-1"],
        search_warnings=["warning"],
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )

    payload = serialize_handler_result(result)

    assert payload == {
        "session_id": "session-1",
        "response_text": "done",
        "phase": "qa",
        "intent": "analyze_paper",
        "referenced_paper_ids": ["paper-1"],
        "artifact_refs": ["paper_workspace:paper-1"],
        "comparison_markdown": None,
        "needs_analysis": False,
        "needs_discovery": False,
        "discovery_topic": None,
        "discovery_candidate_count": None,
        "selected_candidate_ids": ["candidate-1"],
        "search_warnings": ["warning"],
    }


def test_executor_analyze_paper_calls_service_and_serializes_result():
    service = FakeService()
    executor = WorkflowJobExecutor(service)

    result = executor.execute(_job())

    assert service.analyze_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]
    assert result["intent"] == "analyze_paper"
    assert result["artifact_refs"] == ["paper_workspace:1706.03762"]
    assert result["search_warnings"] == ["s2 unavailable"]


def test_executor_analyze_paper_requires_paper_url():
    executor = WorkflowJobExecutor(FakeService())

    with pytest.raises(WorkflowJobExecutionError):
        executor.execute(_job(input_json={}))


def test_executor_analyze_selected_calls_service():
    service = FakeService()
    executor = WorkflowJobExecutor(service)

    result = executor.execute(_job(kind="analyze_selected", input_json={}))

    assert service.analyze_selected_calls == ["session-1"]
    assert result["response_text"] == "selected analysis complete"


def test_executor_rejects_unsupported_kind():
    executor = WorkflowJobExecutor(FakeService())

    with pytest.raises(UnsupportedWorkflowJobKindError):
        executor.execute(_job(kind="compare", input_json={}))


def test_worker_run_once_returns_none_when_idle():
    repository = FakeRepository()
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(FakeService()),
        worker_id="worker-1",
    )

    assert worker.run_once() is None
    assert repository.claim_calls == [{"worker_id": "worker-1", "kinds": None}]


def test_worker_run_once_claims_executes_and_marks_success():
    repository = FakeRepository(jobs=[_job()])
    service = FakeService()
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
        kinds=["analyze_paper"],
    )

    result = worker.run_once()

    assert result is not None
    assert result.status == "succeeded"
    assert repository.claim_calls == [
        {"worker_id": "worker-1", "kinds": ["analyze_paper"]}
    ]
    assert repository.succeeded[0][1]["response_text"] == "analysis complete"
    assert repository.failed == []
    assert repository.mark_running_calls == []


def test_worker_run_once_marks_failure_on_executor_error():
    service = FakeService()
    service.fail_analyze = True
    repository = FakeRepository(jobs=[_job()])
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
    )

    result = worker.run_once()

    assert result is not None
    assert result.status == "failed"
    error = repository.failed[0][1]
    assert error["error"] == "exception"
    assert error["exception_type"] == "RuntimeError"
    assert error["message"] == "analysis down"
    assert error["job_kind"] == "analyze_paper"
    assert repository.succeeded == []


def test_worker_run_once_marks_failure_for_unsupported_kind():
    repository = FakeRepository(jobs=[_job(kind="compare", input_json={})])
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(FakeService()),
        worker_id="worker-1",
    )

    result = worker.run_once()

    assert result is not None
    assert result.status == "failed"
    error = repository.failed[0][1]
    assert error["error"] == "unsupported_job_kind"
    assert error["exception_type"] == "UnsupportedWorkflowJobKindError"
    assert error["job_kind"] == "compare"


def test_worker_run_once_marks_failure_for_invalid_input():
    repository = FakeRepository(jobs=[_job(input_json={})])
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(FakeService()),
        worker_id="worker-1",
    )

    result = worker.run_once()

    assert result is not None
    assert result.status == "failed"
    error = repository.failed[0][1]
    assert error["error"] == "invalid_job_input"
    assert error["exception_type"] == "WorkflowJobExecutionError"
    assert "input_json.paper_url" in error["message"]


def test_worker_run_until_idle_processes_until_no_jobs():
    repository = FakeRepository(jobs=[_job(), _job(kind="analyze_selected", input_json={})])
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(FakeService()),
        worker_id="worker-1",
    )

    assert worker.run_until_idle() == 2
    assert len(repository.succeeded) == 2


def test_serialize_exception_includes_job_context():
    job = _job()

    payload = serialize_exception(RuntimeError("boom"), job=job)

    assert payload["error"] == "exception"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "boom"
    assert payload["job_kind"] == "analyze_paper"
    assert payload["job_id"] == job.id
    assert payload["session_id"] == "session-1"
