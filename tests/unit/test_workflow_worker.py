from dataclasses import dataclass, field
import time

import pytest

from models.jobs import WorkflowJob
from models.registered_pdf_errors import (
    RegisteredPdfBlobNotAuthorizedError,
    RegisteredPdfBlobNotFoundError,
)
from models.session import HandlerResult
from services.blob_store import BlobStoreUnavailableError
from storage.repositories import WorkflowJobLeaseLostError
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
        self.analyze_registered_pdf_blob_calls = []
        self.registered_pdf_blob_error = None
        self.fail_analyze = False
        self.analyze_error = None
        self.analyze_delay_seconds = 0

    def analyze_paper(self, session_id, paper_url):
        self.analyze_calls.append((session_id, paper_url))
        if self.analyze_error is not None:
            raise self.analyze_error
        if self.analyze_delay_seconds:
            time.sleep(self.analyze_delay_seconds)
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

    def analyze_registered_pdf_blob(
        self,
        session_id,
        blob_id,
        *,
        upload_id=None,
        paper_id=None,
        skip_arxiv_metadata_fetch=False,
        pipeline_version="v1",
        cancellation_callback=None,
    ):
        if self.registered_pdf_blob_error is not None:
            raise self.registered_pdf_blob_error
        self.analyze_registered_pdf_blob_calls.append(
            (
                session_id,
                blob_id,
                upload_id,
                paper_id,
                skip_arxiv_metadata_fetch,
                pipeline_version,
            )
        )
        return HandlerResult(
            session_id=session_id,
            response_text="pdf analysis complete",
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
    retries: list[tuple[str, dict]] = field(default_factory=list)
    claim_calls: list[dict] = field(default_factory=list)
    heartbeat_calls: list[dict] = field(default_factory=list)
    mark_running_calls: list[tuple[str, str]] = field(default_factory=list)
    cancel_requested: bool = False
    loaded_job: WorkflowJob | None = None
    lose_lease_on_success: bool = False

    def get(self, job_id):
        return self.loaded_job

    def claim_next(self, *, worker_id, kinds=None, lease_seconds=90):
        self.claim_calls.append(
            {"worker_id": worker_id, "kinds": kinds, "lease_seconds": lease_seconds}
        )
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

    def mark_succeeded(self, job_id, *, worker_id, result_json):
        if self.lose_lease_on_success:
            raise WorkflowJobLeaseLostError(
                job_id=job_id, status="running", target_status="succeeded"
            )
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

    def record_failure(self, job_id, *, worker_id, error_json, retryable):
        target = self.retries if retryable else self.failed
        target.append((job_id, error_json))
        return WorkflowJob(
            id=job_id,
            session_id="session-1",
            kind="analyze_paper",
            status="failed",
            input_json={},
            error_json=error_json,
            attempts=1,
        )

    def heartbeat(self, job_id, *, worker_id, lease_seconds):
        self.heartbeat_calls.append(
            {"job_id": job_id, "worker_id": worker_id, "lease_seconds": lease_seconds}
        )

    def is_cancel_requested(self, job_id, *, worker_id):
        return self.cancel_requested

    def complete_canceled(self, job_id, *, worker_id):
        return WorkflowJob(
            id=job_id,
            session_id="session-1",
            kind="analyze_paper",
            status="canceled",
            input_json={},
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
        "metadata": {},
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


def test_executor_analyze_paper_serializes_reused_analysis_result():
    class ReusedAnalysisService(FakeService):
        def analyze_paper(self, session_id, paper_url):
            self.analyze_calls.append((session_id, paper_url))
            return HandlerResult(
                session_id=session_id,
                response_text="# Cached Report",
                phase="qa",
                intent="analyze_paper",
                referenced_paper_ids=["1706.03762"],
                artifact_refs=["paper_workspace:workspace-cache-hit"],
                needs_analysis=False,
                needs_discovery=False,
                metadata={
                    "analysis_reused": True,
                    "reuse_source": "paper_id",
                },
                user_turn_id="user-turn-cache",
                assistant_turn_id="assistant-turn-cache",
            )

    service = ReusedAnalysisService()
    executor = WorkflowJobExecutor(service)

    result = executor.execute(_job())

    assert service.analyze_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]
    assert result == {
        "session_id": "session-1",
        "response_text": "# Cached Report",
        "phase": "qa",
        "intent": "analyze_paper",
        "referenced_paper_ids": ["1706.03762"],
        "artifact_refs": ["paper_workspace:workspace-cache-hit"],
        "comparison_markdown": None,
        "needs_analysis": False,
        "needs_discovery": False,
        "discovery_topic": None,
        "discovery_candidate_count": None,
        "selected_candidate_ids": [],
        "search_warnings": [],
        "metadata": {
            "analysis_reused": True,
            "reuse_source": "paper_id",
        },
    }


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


def test_executor_analyze_pdf_blob_calls_registered_blob_helper():
    service = FakeService()
    executor = WorkflowJobExecutor(service)

    result = executor.execute(
        _job(
            kind="analyze_pdf_blob",
            input_json={
                "blob_id": "blob-1",
                "upload_id": "upload-1",
                "paper_id": "paper-1",
                "skip_arxiv_metadata_fetch": True,
                "pipeline_version": "v1",
            },
        )
    )

    assert result["response_text"] == "pdf analysis complete"
    assert service.analyze_registered_pdf_blob_calls == [
        ("session-1", "blob-1", "upload-1", "paper-1", True, "v1")
    ]


def test_executor_analyze_pdf_blob_rejects_pipeline_version_mismatch():
    executor = WorkflowJobExecutor(FakeService())

    with pytest.raises(WorkflowJobExecutionError, match="must match"):
        executor.execute(
            WorkflowJob(
                session_id="session-1",
                kind="analyze_pdf_blob",
                pipeline_version="v2",
                input_json={
                    "blob_id": "blob-1",
                    "upload_id": "upload-1",
                    "paper_id": None,
                    "skip_arxiv_metadata_fetch": False,
                    "pipeline_version": "v1",
                },
            )
        )


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
    assert repository.claim_calls == [
        {"worker_id": "worker-1", "kinds": None, "lease_seconds": 90}
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lease_seconds": 0}, "lease_seconds must be positive"),
        (
            {"heartbeat_interval_seconds": 0},
            "heartbeat_interval_seconds must be positive",
        ),
        (
            {"lease_seconds": 30, "heartbeat_interval_seconds": 30},
            "heartbeat_interval_seconds must be less than lease_seconds",
        ),
    ],
)
def test_worker_rejects_invalid_reliability_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorkflowWorker(
            repository=FakeRepository(),
            executor=WorkflowJobExecutor(FakeService()),
            worker_id="worker-1",
            **kwargs,
        )


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
        {"worker_id": "worker-1", "kinds": ["analyze_paper"], "lease_seconds": 90}
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


def test_worker_run_once_records_structured_pdf_blob_failure():
    service = FakeService()
    service.registered_pdf_blob_error = RegisteredPdfBlobNotAuthorizedError(
        session_id="session-1", blob_id="blob-1"
    )
    repository = FakeRepository(
        jobs=[
            _job(
                kind="analyze_pdf_blob",
                input_json={
                    "blob_id": "blob-1",
                    "upload_id": "upload-1",
                    "paper_id": None,
                    "skip_arxiv_metadata_fetch": False,
                    "pipeline_version": "v1",
                },
            )
        ]
    )

    result = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
    ).run_once()

    assert result is not None
    assert result.status == "failed"
    assert repository.failed[0][1]["error"] == "registered_blob_not_authorized"


def test_worker_run_once_completes_requested_cancel_before_execution():
    repository = FakeRepository(jobs=[_job()], cancel_requested=True)
    service = FakeService()

    result = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
    ).run_once()

    assert result.status == "canceled"
    assert service.analyze_calls == []
    assert repository.succeeded == []
    assert repository.failed == []


def test_worker_run_once_records_retryable_blob_store_failure():
    service = FakeService()
    service.analyze_error = BlobStoreUnavailableError("blob store down")
    repository = FakeRepository(jobs=[_job()])

    WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
    ).run_once()

    assert repository.retries[0][1]["message"] == "blob store down"
    assert repository.failed == []


def test_worker_run_once_heartbeats_during_execution():
    service = FakeService()
    service.analyze_delay_seconds = 0.03
    repository = FakeRepository(jobs=[_job()])

    WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id="worker-1",
        heartbeat_interval_seconds=0.005,
    ).run_once()

    assert repository.heartbeat_calls


def test_worker_run_once_returns_current_job_after_stale_success_write():
    current = WorkflowJob(
        id="job-1",
        session_id="session-1",
        kind="analyze_paper",
        status="running",
        input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
        locked_by="worker-2",
        attempts=2,
    )
    repository = FakeRepository(
        jobs=[
            WorkflowJob(
                id=current.id,
                session_id=current.session_id,
                kind=current.kind,
                input_json=current.input_json,
            )
        ],
        loaded_job=current,
        lose_lease_on_success=True,
    )

    result = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(FakeService()),
        worker_id="worker-1",
    ).run_once()

    assert result == current
    assert repository.succeeded == []


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


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (RegisteredPdfBlobNotFoundError("blob-1"), "blob_not_found"),
        (
            RegisteredPdfBlobNotAuthorizedError(
                session_id="session-1", blob_id="blob-1"
            ),
            "registered_blob_not_authorized",
        ),
        (RuntimeError("analysis down"), "analysis_failed"),
    ],
)
def test_serialize_exception_maps_pdf_blob_failures(exc, error):
    job = _job(
        kind="analyze_pdf_blob",
        input_json={
            "blob_id": "blob-1",
            "upload_id": "upload-1",
            "paper_id": None,
            "skip_arxiv_metadata_fetch": False,
            "pipeline_version": "v1",
        },
    )

    assert serialize_exception(exc, job=job)["error"] == error
