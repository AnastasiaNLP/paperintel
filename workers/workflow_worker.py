from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Protocol

import httpx

from agents.cancellation import WorkflowCancellationRequested
from models.jobs import JobKind, WorkflowJob
from models.pdf_upload_errors import PdfUploadStateError
from models.registered_pdf_errors import (
    RegisteredPdfBlobNotAuthorizedError,
    RegisteredPdfBlobNotFoundError,
)
from models.session import HandlerResult
from services.blob_store import (
    BlobIntegrityError,
    BlobNotFoundError,
    BlobSizeLimitError,
    BlobStoreUnavailableError,
)
from services.paperintel_service import PaperIntelService
from services.provider_policy import classify_provider_exception
from services.qdrant_store import QdrantDependencyError
from storage.repositories import WorkflowJobLeaseLostError
from tools.circuit_breaker import CircuitBreakerOpenError

LOGGER = logging.getLogger(__name__)
DEFAULT_LEASE_SECONDS = 90
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
SUPPORTED_JOB_KINDS: set[JobKind] = {
    "analyze_paper",
    "analyze_selected",
    "analyze_pdf_blob",
}


class WorkflowJobRepository(Protocol):
    def get(self, job_id: str) -> WorkflowJob | None:
        ...

    def claim_next(
        self,
        *,
        worker_id: str,
        kinds: list[JobKind] | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> WorkflowJob | None:
        ...

    def mark_succeeded(
        self, job_id: str, *, worker_id: str, result_json: dict
    ) -> WorkflowJob:
        ...

    def record_failure(
        self,
        job_id: str,
        *,
        worker_id: str,
        error_json: dict,
        retryable: bool,
        retry_after_seconds: float | None = None,
    ) -> WorkflowJob:
        ...

    def heartbeat(
        self, job_id: str, *, worker_id: str, lease_seconds: int
    ) -> WorkflowJob:
        ...

    def is_cancel_requested(self, job_id: str, *, worker_id: str) -> bool:
        ...

    def complete_canceled(self, job_id: str, *, worker_id: str) -> WorkflowJob:
        ...


class WorkflowJobExecutionError(ValueError):
    pass


class UnsupportedWorkflowJobKindError(WorkflowJobExecutionError):
    def __init__(self, kind: str) -> None:
        super().__init__(f"Unsupported workflow job kind: {kind}")
        self.kind = kind


class WorkflowJobExecutor:
    def __init__(self, service: PaperIntelService) -> None:
        self.service = service

    def execute(self, job: WorkflowJob, *, cancellation_callback=None) -> dict:
        if job.kind == "analyze_paper":
            return self._execute_analyze_paper(job)
        if job.kind == "analyze_selected":
            return self._execute_analyze_selected(job)
        if job.kind == "analyze_pdf_blob":
            return self._execute_analyze_pdf_blob(
                job, cancellation_callback=cancellation_callback
            )
        raise UnsupportedWorkflowJobKindError(job.kind)

    def _execute_analyze_paper(self, job: WorkflowJob) -> dict:
        paper_url = job.input_json.get("paper_url")
        if not isinstance(paper_url, str) or not paper_url.strip():
            raise WorkflowJobExecutionError(
                "analyze_paper job requires non-empty input_json.paper_url"
            )
        result = self.service.analyze_paper(job.session_id, paper_url.strip())
        return serialize_handler_result(result)

    def _execute_analyze_selected(self, job: WorkflowJob) -> dict:
        result = self.service.analyze_selected_papers(job.session_id)
        return serialize_handler_result(result)

    def _execute_analyze_pdf_blob(self, job: WorkflowJob, *, cancellation_callback=None) -> dict:
        blob_id = job.input_json.get("blob_id")
        upload_id = job.input_json.get("upload_id")
        paper_id = job.input_json.get("paper_id")
        skip_arxiv_metadata_fetch = job.input_json.get(
            "skip_arxiv_metadata_fetch", False
        )
        pipeline_version = job.input_json.get("pipeline_version")
        if not isinstance(blob_id, str) or not blob_id.strip():
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job requires non-empty input_json.blob_id"
            )
        if not isinstance(upload_id, str) or not upload_id.strip():
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job requires non-empty input_json.upload_id"
            )
        if paper_id is not None and not isinstance(paper_id, str):
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job input_json.paper_id must be a string or null"
            )
        if not isinstance(skip_arxiv_metadata_fetch, bool):
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job input_json.skip_arxiv_metadata_fetch must be a boolean"
            )
        if not isinstance(pipeline_version, str) or not pipeline_version.strip():
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job requires non-empty input_json.pipeline_version"
            )
        if pipeline_version.strip() != job.pipeline_version:
            raise WorkflowJobExecutionError(
                "analyze_pdf_blob job input_json.pipeline_version must match job.pipeline_version"
            )
        result = self.service.analyze_registered_pdf_blob(
            job.session_id,
            blob_id.strip(),
            upload_id=upload_id.strip(),
            paper_id=paper_id.strip() if paper_id is not None else None,
            skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
            pipeline_version=job.pipeline_version,
            cancellation_callback=cancellation_callback,
        )
        return serialize_handler_result(result)


class WorkflowWorker:
    def __init__(
        self,
        *,
        repository: WorkflowJobRepository,
        executor: WorkflowJobExecutor,
        worker_id: str,
        poll_interval: float = 2.0,
        kinds: list[JobKind] | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive.")
        if heartbeat_interval_seconds >= lease_seconds:
            raise ValueError("heartbeat_interval_seconds must be less than lease_seconds.")
        self.repository = repository
        self.executor = executor
        self.worker_id = worker_id
        self.poll_interval = poll_interval
        self.kinds = kinds
        self.lease_seconds = lease_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    def run_once(self) -> WorkflowJob | None:
        job = self.repository.claim_next(
            worker_id=self.worker_id,
            kinds=self.kinds,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return None
        if job.status != "running":
            return job

        LOGGER.info(
            "Workflow job claimed: id=%s kind=%s worker_id=%s",
            job.id,
            job.kind,
            self.worker_id,
        )
        cancellation_callback = lambda: self._raise_if_canceled(job.id)
        try:
            cancellation_callback()
            with _heartbeat_runner(
                repository=self.repository,
                job_id=job.id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_interval_seconds,
            ) as heartbeat_errors:
                result_json = self.executor.execute(
                    job, cancellation_callback=cancellation_callback
                )
                cancellation_callback()
                if heartbeat_errors:
                    raise heartbeat_errors[0]
        except WorkflowJobCanceledError:
            try:
                canceled = self.repository.complete_canceled(
                    job.id, worker_id=self.worker_id
                )
            except WorkflowJobLeaseLostError as exc:
                return self._load_after_lease_loss(job, exc)
            LOGGER.info(
                "Workflow job canceled: id=%s kind=%s worker_id=%s",
                job.id,
                job.kind,
                self.worker_id,
            )
            return canceled
        except WorkflowJobLeaseLostError as exc:
            return self._load_after_lease_loss(job, exc)
        except Exception as exc:
            retry_decision = decide_workflow_retry(exc)
            will_retry = retry_decision.retryable and job.attempts < job.max_attempts
            try:
                failed = self.repository.record_failure(
                    job.id,
                    worker_id=self.worker_id,
                    error_json=serialize_exception(
                        exc,
                        job=job,
                        retry_decision=retry_decision,
                        will_retry=will_retry,
                    ),
                    retryable=retry_decision.retryable,
                    retry_after_seconds=retry_decision.retry_after_seconds,
                )
            except WorkflowJobLeaseLostError as lease_exc:
                return self._load_after_lease_loss(job, lease_exc)
            LOGGER.exception(
                (
                    "Workflow job failed: id=%s kind=%s worker_id=%s "
                    "failure_class=%s retryable=%s retry_after_seconds=%s "
                    "attempts=%s/%s"
                ),
                job.id,
                job.kind,
                self.worker_id,
                retry_decision.failure_class,
                will_retry,
                retry_decision.retry_after_seconds,
                job.attempts,
                job.max_attempts,
            )
            return failed

        try:
            succeeded = self.repository.mark_succeeded(
                job.id, worker_id=self.worker_id, result_json=result_json
            )
        except WorkflowJobLeaseLostError as exc:
            return self._load_after_lease_loss(job, exc)
        if succeeded.status == "canceled":
            LOGGER.info(
                "Workflow job canceled before success commit: id=%s kind=%s worker_id=%s",
                job.id,
                job.kind,
                self.worker_id,
            )
            return succeeded
        LOGGER.info(
            "Workflow job succeeded: id=%s kind=%s worker_id=%s",
            job.id,
            job.kind,
            self.worker_id,
        )
        return succeeded

    def run_until_idle(self, *, max_jobs: int | None = None) -> int:
        processed = 0
        while max_jobs is None or processed < max_jobs:
            job = self.run_once()
            if job is None:
                return processed
            processed += 1
        return processed

    def run_forever(self) -> None:
        while True:
            job = self.run_once()
            if job is None:
                time.sleep(self.poll_interval)

    def _raise_if_canceled(self, job_id: str) -> None:
        if self.repository.is_cancel_requested(job_id, worker_id=self.worker_id):
            raise WorkflowJobCanceledError(job_id)

    def _load_after_lease_loss(
        self, job: WorkflowJob, exc: WorkflowJobLeaseLostError
    ) -> WorkflowJob | None:
        LOGGER.warning(
            "Workflow job lease lost: id=%s kind=%s worker_id=%s error=%s",
            job.id,
            job.kind,
            self.worker_id,
            exc,
        )
        return self.repository.get(job.id)


class WorkflowJobCanceledError(WorkflowCancellationRequested):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Workflow job cancellation requested: {job_id}")
        self.job_id = job_id


@contextmanager
def _heartbeat_runner(
    *,
    repository: WorkflowJobRepository,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    interval_seconds: float,
):
    stop = Event()
    errors: list[Exception] = []

    def heartbeat_loop() -> None:
        while not stop.wait(interval_seconds):
            try:
                repository.heartbeat(
                    job_id, worker_id=worker_id, lease_seconds=lease_seconds
                )
            except Exception as exc:
                errors.append(exc)
                return

    thread = Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        yield errors
    finally:
        stop.set()
        thread.join()


def serialize_handler_result(result: HandlerResult) -> dict:
    return {
        "session_id": result.session_id,
        "response_text": result.response_text,
        "phase": result.phase,
        "intent": result.intent,
        "referenced_paper_ids": list(result.referenced_paper_ids),
        "artifact_refs": list(result.artifact_refs),
        "comparison_markdown": result.comparison_markdown,
        "needs_analysis": result.needs_analysis,
        "needs_discovery": result.needs_discovery,
        "discovery_topic": result.discovery_topic,
        "discovery_candidate_count": result.discovery_candidate_count,
        "selected_candidate_ids": list(result.selected_candidate_ids),
        "search_warnings": list(result.search_warnings),
        "metadata": dict(result.metadata),
    }


@dataclass(frozen=True)
class WorkflowRetryDecision:
    retryable: bool
    failure_class: str
    retry_after_seconds: float | None = None


def serialize_exception(
    exc: Exception,
    *,
    job: WorkflowJob,
    retry_decision: WorkflowRetryDecision | None = None,
    will_retry: bool | None = None,
) -> dict:
    classified = classify_workflow_failure(exc)
    decision = retry_decision or decide_workflow_retry(exc)
    actual_retryable = decision.retryable if will_retry is None else will_retry
    error = "exception"
    if isinstance(exc, UnsupportedWorkflowJobKindError):
        error = "unsupported_job_kind"
    elif isinstance(exc, WorkflowJobExecutionError):
        error = "invalid_job_input"
    elif isinstance(exc, (RegisteredPdfBlobNotFoundError, BlobNotFoundError)):
        error = "blob_not_found"
    elif isinstance(exc, (BlobIntegrityError, BlobSizeLimitError)):
        error = "blob_integrity_failure"
    elif isinstance(exc, PdfUploadStateError):
        error = "pdf_upload_not_ready"
    elif isinstance(exc, RegisteredPdfBlobNotAuthorizedError):
        error = "registered_blob_not_authorized"
    elif job.kind == "analyze_pdf_blob":
        error = "analysis_failed"
    return {
        "error": error,
        "failure_class": classified.failure_class.value,
        "retryable": actual_retryable,
        "exception_type": type(exc).__name__,
        "message": _safe_exception_message(exc),
        "job_kind": job.kind,
        "job_id": job.id,
        "session_id": job.session_id,
        **(
            {"retry_after_seconds": decision.retry_after_seconds}
            if actual_retryable and decision.retry_after_seconds is not None
            else {}
        ),
    }


def is_retryable_failure(exc: Exception) -> bool:
    return decide_workflow_retry(exc).retryable


def decide_workflow_retry(exc: Exception) -> WorkflowRetryDecision:
    classified = classify_workflow_failure(exc)
    retry_after_seconds = _retry_after_seconds_from_exception(exc)
    return WorkflowRetryDecision(
        retryable=classified.retryable or isinstance(exc, CircuitBreakerOpenError),
        failure_class=classified.failure_class.value,
        retry_after_seconds=retry_after_seconds,
    )


def classify_workflow_failure(exc: Exception):
    return classify_provider_exception(
        "workflow",
        "job_execution",
        exc,
        not_found_exception_types=(RegisteredPdfBlobNotFoundError,),
        invalid_input_exception_types=(
            UnsupportedWorkflowJobKindError,
            WorkflowJobExecutionError,
            BlobIntegrityError,
            BlobSizeLimitError,
            PdfUploadStateError,
            RegisteredPdfBlobNotAuthorizedError,
        ),
        dependency_unavailable_exception_types=(
            BlobStoreUnavailableError,
            QdrantDependencyError,
        ),
        dependency_not_found_exception_types=(BlobNotFoundError,),
        circuit_open_exception_types=(CircuitBreakerOpenError,),
        canceled_exception_types=(WorkflowCancellationRequested,),
    )


def _safe_exception_message(exc: Exception) -> str:
    if isinstance(exc, CircuitBreakerOpenError):
        return "Provider is temporarily unavailable"
    return str(exc)


def _retry_after_seconds_from_exception(exc: Exception) -> float | None:
    if isinstance(exc, CircuitBreakerOpenError):
        return exc.retry_after_seconds
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return _parse_retry_after_header(exc.response.headers.get("Retry-After"))
    return None


def _parse_retry_after_header(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(seconds, 0.0)
