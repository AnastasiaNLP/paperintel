from __future__ import annotations

import logging
import time
from typing import Protocol

from models.jobs import JobKind, WorkflowJob
from models.pdf_upload_errors import PdfUploadStateError
from models.registered_pdf_errors import (
    RegisteredPdfBlobNotAuthorizedError,
    RegisteredPdfBlobNotFoundError,
)
from models.session import HandlerResult
from services.blob_store import BlobIntegrityError, BlobNotFoundError, BlobSizeLimitError
from services.paperintel_service import PaperIntelService

LOGGER = logging.getLogger(__name__)
SUPPORTED_JOB_KINDS: set[JobKind] = {
    "analyze_paper",
    "analyze_selected",
    "analyze_pdf_blob",
}


class WorkflowJobRepository(Protocol):
    def claim_next(
        self,
        *,
        worker_id: str,
        kinds: list[JobKind] | None = None,
    ) -> WorkflowJob | None:
        ...

    def mark_succeeded(self, job_id: str, *, result_json: dict) -> WorkflowJob:
        ...

    def mark_failed(self, job_id: str, *, error_json: dict) -> WorkflowJob:
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

    def execute(self, job: WorkflowJob) -> dict:
        if job.kind == "analyze_paper":
            return self._execute_analyze_paper(job)
        if job.kind == "analyze_selected":
            return self._execute_analyze_selected(job)
        if job.kind == "analyze_pdf_blob":
            return self._execute_analyze_pdf_blob(job)
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

    def _execute_analyze_pdf_blob(self, job: WorkflowJob) -> dict:
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
    ) -> None:
        self.repository = repository
        self.executor = executor
        self.worker_id = worker_id
        self.poll_interval = poll_interval
        self.kinds = kinds

    def run_once(self) -> WorkflowJob | None:
        job = self.repository.claim_next(worker_id=self.worker_id, kinds=self.kinds)
        if job is None:
            return None

        LOGGER.info(
            "Workflow job claimed: id=%s kind=%s worker_id=%s",
            job.id,
            job.kind,
            self.worker_id,
        )
        try:
            result_json = self.executor.execute(job)
        except Exception as exc:
            failed = self.repository.mark_failed(
                job.id,
                error_json=serialize_exception(exc, job=job),
            )
            LOGGER.exception(
                "Workflow job failed: id=%s kind=%s worker_id=%s",
                job.id,
                job.kind,
                self.worker_id,
            )
            return failed

        succeeded = self.repository.mark_succeeded(job.id, result_json=result_json)
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
    }


def serialize_exception(exc: Exception, *, job: WorkflowJob) -> dict:
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
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "job_kind": job.kind,
        "job_id": job.id,
        "session_id": job.session_id,
    }
