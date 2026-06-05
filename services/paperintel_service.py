import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agents.comparison_analyst import compare_workspaces
from agents.synthesis_agent import synthesize_workspaces
from api.chat_handler import ChatHandler
from models.api import HealthStatus
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_artifacts import BlobArtifact, BlobReference, BlobReferenceKind
from models.blob_storage import StoredBlobObject
from models.discovery import CandidateStatus, SearchCandidate
from models.jobs import WorkflowJob
from models.pdf_upload_errors import (
    PdfUploadChecksumMismatchError,
    PdfUploadExpiredError,
    PdfUploadInvalidContentError,
    PdfUploadNotFoundError,
    PdfUploadSizeMismatchError,
    PdfUploadStateError,
)
from models.pdf_uploads import PdfUpload, PdfUploadInitiation
from models.registered_pdf_errors import (
    RegisteredPdfBlobNotAuthorizedError,
    RegisteredPdfBlobNotFoundError,
)
from models.retrieval import PaperChunk, UpsertChunksResult
from models.session import utc_now
from models.session import HandlerResult, Persona, Session, Turn
from models.synthesis import SynthesisAgentResult
from services.blob_store import BlobSizeLimitError, BlobStore, BlobStoreUnavailableError
from services.selected_candidate_resolver import SelectedCandidateResolver

_FAILED_WORKSPACE_STAGES = {"failed", "paper_failure_finalize"}
MAX_LOCAL_PDF_BYTES = 50 * 1024 * 1024
MIN_UPLOAD_URL_EXPIRES_SECONDS = 60
MAX_UPLOAD_URL_EXPIRES_SECONDS = 3600


class InvalidPdfInputError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidSessionPhaseError(ValueError):
    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            f"Session is not in {expected} phase; current phase is {actual}."
        )
        self.expected = expected
        self.actual = actual


class NoActivePapersError(ValueError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} has no active papers. Analyze papers before synthesis."
        )
        self.session_id = session_id


class PaperWorkspaceNotFoundError(ValueError):
    def __init__(self, *, session_id: str, paper_id: str) -> None:
        super().__init__(
            f"Paper workspace {paper_id} was not found in session {session_id}."
        )
        self.session_id = session_id
        self.paper_id = paper_id


class PaperWorkspaceNotReadyError(ValueError):
    def __init__(
        self,
        *,
        session_id: str,
        paper_id: str,
        pipeline_stage: str,
    ) -> None:
        super().__init__(
            f"Paper workspace {paper_id} in session {session_id} is not ready "
            f"for request-driven comparison or synthesis; stage={pipeline_stage}."
        )
        self.session_id = session_id
        self.paper_id = paper_id
        self.pipeline_stage = pipeline_stage


class ComparisonNotFoundError(ValueError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"No comparison artifact was found for session {session_id}.")
        self.session_id = session_id


class NotEnoughPapersForComparisonError(ValueError):
    def __init__(self, *, session_id: str, paper_ids: list[str]) -> None:
        super().__init__(
            f"Comparison requires at least two papers in session {session_id}; "
            f"got {len(paper_ids)}."
        )
        self.session_id = session_id
        self.paper_ids = paper_ids


class BlobStorageNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Blob storage requires both blob_store and blob_artifact_repository."
        )


class PdfUploadNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF uploads require blob storage and pdf_upload_repository.")


class PaperCacheHydrationNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Reusable analysis is not available.")


class PaperCacheHydrationEmptyChunksError(RuntimeError):
    def __init__(self, *, session_id: str, paper_id: str) -> None:
        super().__init__("Reusable analysis is incomplete.")
        self.session_id = session_id
        self.paper_id = paper_id



class WorkflowJobNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Workflow job repository is not configured.")


class WorkflowJobNotFoundError(ValueError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Workflow job not found: {job_id}")
        self.job_id = job_id


class InvalidWorkflowJobInputError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _validate_local_pdf_path(pdf_path: str) -> str:
    path = Path(pdf_path).expanduser()
    if not str(pdf_path).strip():
        raise InvalidPdfInputError("pdf_path must not be empty")
    if not path.exists():
        raise InvalidPdfInputError(f"PDF file does not exist: {pdf_path}")
    if not path.is_file():
        raise InvalidPdfInputError(f"PDF path is not a file: {pdf_path}")
    if path.stat().st_size > MAX_LOCAL_PDF_BYTES:
        raise InvalidPdfInputError(
            f"PDF file is too large; max size is {MAX_LOCAL_PDF_BYTES} bytes"
        )
    with path.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise InvalidPdfInputError("PDF file must start with %PDF- magic bytes")
    return str(path)


class SearchCandidateRepository(Protocol):
    def update_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
    ) -> SearchCandidate | None:
        ...


class PaperWorkspaceRepository(Protocol):
    def list_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        ...

    def get_workspace(
        self,
        session_id: str,
        paper_id: str,
    ) -> PaperWorkspace | None:
        ...

    def find_reusable_workspace(
        self,
        *,
        paper_id: str,
        pipeline_version: str,
        exclude_session_id: str | None = None,
    ) -> PaperWorkspace | None:
        ...

    def find_reusable_workspace_by_pdf_hash(
        self,
        *,
        content_hash: str,
        pipeline_version: str,
        exclude_session_id: str | None = None,
    ) -> PaperWorkspace | None:
        ...

    def clone_workspace(
        self,
        *,
        source_workspace_id: str,
        target_session_id: str,
    ) -> PaperWorkspace:
        ...

    def latest_comparison(self, session_id: str) -> ComparisonArtifact | None:
        ...

    def save_comparison(self, artifact: ComparisonArtifact) -> ComparisonArtifact:
        ...


class PaperChunkRepository(Protocol):
    def upsert_many(self, chunks: list[PaperChunk]) -> UpsertChunksResult:
        ...

    def list_for_session_paper(self, session_id: str, paper_id: str) -> list[PaperChunk]:
        ...

    def clone_for_session(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        paper_id: str,
        target_paper_id: str | None = None,
    ) -> list[PaperChunk]:
        ...


class BlobArtifactRepository(Protocol):
    def upsert_artifact(self, stored: StoredBlobObject, **kwargs) -> BlobArtifact:
        ...

    def add_reference(
        self,
        blob_id: str,
        *,
        ref_kind: BlobReferenceKind,
        ref_id: str,
        metadata: dict | None = None,
    ) -> BlobReference:
        ...

    def mark_accessed(self, blob_id: str) -> BlobArtifact:
        ...

    def get_artifact(self, blob_id: str) -> BlobArtifact | None:
        ...

    def has_active_reference(
        self, blob_id: str, *, ref_kind: BlobReferenceKind, ref_id: str
    ) -> bool:
        ...


class PdfUploadRepository(Protocol):
    def create(self, upload: PdfUpload) -> PdfUpload:
        ...

    def get(self, upload_id: str) -> PdfUpload | None:
        ...

    def mark_uploaded(self, upload_id: str) -> PdfUpload:
        ...

    def finalize(
        self,
        upload_id: str,
        *,
        blob_id: str,
        actual_sha256: str,
        size_bytes: int,
    ) -> PdfUpload:
        ...

    def mark_failed(self, upload_id: str, *, error_json: dict) -> PdfUpload:
        ...


class WorkflowJobRepository(Protocol):
    def create(self, job: WorkflowJob) -> WorkflowJob:
        ...

    def get(self, job_id: str) -> WorkflowJob | None:
        ...

    def list_for_session(self, session_id: str, limit: int = 50) -> list[WorkflowJob]:
        ...

    def mark_canceled(self, job_id: str) -> WorkflowJob:
        ...

    def enqueue_pdf_blob(
        self,
        *,
        session_id: str,
        upload_id: str,
        paper_id: str | None,
        skip_arxiv_metadata_fetch: bool,
        pipeline_version: str,
    ) -> WorkflowJob:
        ...


class PaperIntelService:
    """
    Product-facing application facade for PaperIntel.

    Transport adapters should depend on this service instead of touching
    ChatHandler, graphs, or storage directly.
    """

    def __init__(
        self,
        *,
        handler: ChatHandler,
        health_checker=None,
        selected_candidate_resolver: SelectedCandidateResolver | None = None,
        candidate_repository: SearchCandidateRepository | None = None,
        artifact_repository: PaperWorkspaceRepository | None = None,
        workflow_job_repository: WorkflowJobRepository | None = None,
        paper_chunk_repository: PaperChunkRepository | None = None,
        blob_store: BlobStore | None = None,
        blob_artifact_repository: BlobArtifactRepository | None = None,
        pdf_upload_repository: PdfUploadRepository | None = None,
    ) -> None:
        self.handler = handler
        self.health_checker = health_checker
        self.selected_candidate_resolver = selected_candidate_resolver
        self.candidate_repository = candidate_repository
        self.artifact_repository = artifact_repository
        self.workflow_job_repository = workflow_job_repository
        self.paper_chunk_repository = paper_chunk_repository
        self.blob_store = blob_store
        self.blob_artifact_repository = blob_artifact_repository
        self.pdf_upload_repository = pdf_upload_repository

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        return self.handler.create_session(
            persona=persona,
            original_query=original_query,
        )

    def handle_message(self, session_id: str, message: str) -> HandlerResult:
        return self.handler.handle_message(session_id, message)

    def analyze_paper(self, session_id: str, paper_url: str) -> HandlerResult:
        return self.handler.handle_message(session_id, paper_url)

    def enqueue_analyze_paper(self, session_id: str, paper_url: str) -> WorkflowJob:
        self.handler.store.require_session(session_id)
        paper_url = paper_url.strip() if isinstance(paper_url, str) else ""
        if not paper_url:
            raise InvalidWorkflowJobInputError("paper_url must not be empty")
        return self._workflow_jobs().create(
            WorkflowJob(
                session_id=session_id,
                kind="analyze_paper",
                input_json={"paper_url": paper_url},
            )
        )

    def enqueue_analyze_selected(self, session_id: str) -> WorkflowJob:
        self.handler.store.require_session(session_id)
        return self._workflow_jobs().create(
            WorkflowJob(
                session_id=session_id,
                kind="analyze_selected",
                input_json={},
            )
        )

    def enqueue_analyze_pdf_blob(
        self,
        session_id: str,
        upload_id: str,
        *,
        paper_id: str | None = None,
        skip_arxiv_metadata_fetch: bool = False,
        pipeline_version: str = "v1",
    ) -> WorkflowJob:
        self.handler.store.require_session(session_id)
        upload_id = upload_id.strip() if isinstance(upload_id, str) else ""
        paper_id = (paper_id.strip() or None) if isinstance(paper_id, str) else None
        pipeline_version = (
            pipeline_version.strip() if isinstance(pipeline_version, str) else ""
        )
        if not upload_id:
            raise InvalidWorkflowJobInputError("upload_id must not be empty")
        if not pipeline_version:
            raise InvalidWorkflowJobInputError("pipeline_version must not be empty")
        return self._workflow_jobs().enqueue_pdf_blob(
            session_id=session_id,
            upload_id=upload_id,
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=bool(skip_arxiv_metadata_fetch),
            pipeline_version=pipeline_version,
        )

    def get_workflow_job(self, job_id: str) -> WorkflowJob:
        job = self._workflow_jobs().get(job_id)
        if job is None:
            raise WorkflowJobNotFoundError(job_id)
        return job

    def list_workflow_jobs(self, session_id: str, *, limit: int = 50) -> list[WorkflowJob]:
        self.handler.store.require_session(session_id)
        return self._workflow_jobs().list_for_session(session_id, limit=limit)

    def cancel_workflow_job(self, job_id: str) -> WorkflowJob:
        self.get_workflow_job(job_id)
        return self._workflow_jobs().mark_canceled(job_id)

    def analyze_pdf(
        self,
        session_id: str,
        pdf_path: str,
        *,
        paper_id: str | None = None,
        skip_arxiv_metadata_fetch: bool = False,
    ) -> HandlerResult:
        self.handler.store.require_session(session_id)
        resolved_pdf_path = _validate_local_pdf_path(pdf_path)
        if self.blob_store is None and self.blob_artifact_repository is None:
            return self._analyze_pdf_path(
                session_id,
                resolved_pdf_path,
                user_content=f"Analyze local PDF {paper_id or resolved_pdf_path}",
                paper_id=paper_id,
                skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
            )
        if self.blob_store is None or self.blob_artifact_repository is None:
            raise BlobStorageNotConfiguredError()
        return self._analyze_pdf_via_blob_store(
            session_id,
            resolved_pdf_path,
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
        )

    def _analyze_pdf_via_blob_store(
        self,
        session_id: str,
        source_pdf_path: str,
        *,
        paper_id: str | None,
        skip_arxiv_metadata_fetch: bool,
    ) -> HandlerResult:
        artifact = self._store_pdf_blob(
            session_id,
            Path(source_pdf_path).read_bytes(),
            source="pdf_ingestion",
        )
        return self.analyze_registered_pdf_blob(
            session_id,
            artifact.id,
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
            user_content=f"Analyze local PDF {paper_id or source_pdf_path}",
        )

    def initiate_pdf_upload(
        self,
        session_id: str,
        *,
        expected_sha256: str,
        size_bytes: int,
        content_type: str = "application/pdf",
        expires_seconds: int = 900,
    ) -> PdfUploadInitiation:
        self.handler.store.require_session(session_id)
        blob_store, _, upload_repository = self._pdf_upload_dependencies()
        upload = self._new_pdf_upload(
            session_id, expected_sha256=expected_sha256, size_bytes=size_bytes,
            content_type=content_type, expires_seconds=expires_seconds,
        )
        upload_repository.create(upload)
        try:
            upload_url = blob_store.create_presigned_put(
                upload.object_key, content_type=content_type, expires_seconds=expires_seconds
            )
        except BlobStoreUnavailableError as exc:
            upload_repository.mark_failed(
                upload.id, error_json={"code": exc.__class__.__name__, "message": str(exc)}
            )
            raise
        return PdfUploadInitiation(
            upload=upload, upload_url=upload_url,
            upload_headers={"Content-Type": content_type},
        )

    def finalize_pdf_upload(self, session_id: str, upload_id: str) -> PdfUpload:
        self.handler.store.require_session(session_id)
        blob_store, artifact_repository, upload_repository = self._pdf_upload_dependencies()
        upload = upload_repository.get(upload_id)
        if upload is None or upload.session_id != session_id:
            raise PdfUploadNotFoundError(upload_id)
        if upload.status not in {"initiated", "uploaded"}:
            raise PdfUploadStateError(
                upload_id=upload.id, status=upload.status, target_status="finalized"
            )
        if upload.expires_at <= utc_now():
            raise PdfUploadExpiredError(upload.id)
        try:
            metadata = blob_store.head_object(upload.object_key)
            self._validate_pdf_upload_metadata(upload, metadata)
            if upload.status == "initiated":
                upload = upload_repository.mark_uploaded(upload.id)
            with blob_store.materialize(
                upload.object_key, max_bytes=MAX_LOCAL_PDF_BYTES
            ) as staged_path:
                content = Path(staged_path).read_bytes()
            self._validate_pdf_upload_content(upload, content)
            stored = blob_store.put(content, kind="pdf", content_type="application/pdf")
            artifact = artifact_repository.upsert_artifact(stored, retention_policy="durable")
            artifact_repository.add_reference(
                artifact.id, ref_kind="session", ref_id=session_id,
                metadata={"source": "pdf_upload", "upload_id": upload.id},
            )
            finalized = upload_repository.finalize(
                upload.id, blob_id=artifact.id, actual_sha256=stored.content_hash,
                size_bytes=stored.size_bytes,
            )
            try:
                blob_store.delete(upload.object_key)
            except BlobStoreUnavailableError:
                # Expired-upload cleanup will reconcile staging objects after provider recovery.
                pass
            return finalized
        except (
            BlobSizeLimitError,
            PdfUploadChecksumMismatchError,
            PdfUploadInvalidContentError,
            PdfUploadSizeMismatchError,
        ) as exc:
            upload_repository.mark_failed(
                upload.id, error_json={"code": exc.__class__.__name__, "message": str(exc)}
            )
            try:
                blob_store.delete(upload.object_key)
            except BlobStoreUnavailableError:
                # Expired-upload cleanup will reconcile staging objects after provider recovery.
                pass
            raise

    def store_pdf_upload(
        self,
        session_id: str,
        content: bytes,
        *,
        expected_sha256: str | None = None,
    ) -> PdfUpload:
        self.handler.store.require_session(session_id)
        blob_store, _, upload_repository = self._pdf_upload_dependencies()
        digest = expected_sha256 or hashlib.sha256(content).hexdigest()
        upload = self._new_pdf_upload(
            session_id, expected_sha256=digest, size_bytes=len(content),
            content_type="application/pdf", expires_seconds=900,
        )
        upload_repository.create(upload)
        try:
            blob_store.put_staging(upload.object_key, content, content_type="application/pdf")
        except BlobStoreUnavailableError as exc:
            upload_repository.mark_failed(
                upload.id, error_json={"code": exc.__class__.__name__, "message": str(exc)}
            )
            raise
        return self.finalize_pdf_upload(session_id, upload.id)

    def analyze_registered_pdf_blob(
        self,
        session_id: str,
        blob_id: str,
        *,
        upload_id: str | None = None,
        paper_id: str | None = None,
        skip_arxiv_metadata_fetch: bool = False,
        pipeline_version: str = "v1",
        user_content: str | None = None,
        cancellation_callback=None,
    ) -> HandlerResult:
        self.handler.store.require_session(session_id)
        blob_store, artifact_repository = self._blob_dependencies()
        if upload_id is not None:
            upload = self._pdf_upload_repository().get(upload_id)
            if (
                upload is None
                or upload.session_id != session_id
                or upload.status != "enqueued"
                or upload.blob_id != blob_id
            ):
                raise PdfUploadStateError(
                    upload_id=upload_id,
                    status=upload.status if upload is not None else "missing",
                    target_status="analyze",
                )
        artifact = artifact_repository.get_artifact(blob_id)
        if artifact is None or artifact.kind != "pdf":
            raise RegisteredPdfBlobNotFoundError(blob_id)
        if not artifact_repository.has_active_reference(
            blob_id, ref_kind="session", ref_id=session_id
        ):
            raise RegisteredPdfBlobNotAuthorizedError(
                session_id=session_id, blob_id=blob_id
            )
        before_workspace_ids = self._workspace_ids(session_id)
        with blob_store.materialize(
            artifact.object_key,
            expected_sha256=artifact.content_hash,
            max_bytes=MAX_LOCAL_PDF_BYTES,
        ) as materialized_path:
            artifact_repository.mark_accessed(artifact.id)
            result = self._analyze_pdf_path(
                session_id, materialized_path,
                user_content=user_content or f"Analyze registered PDF blob {blob_id}",
                paper_id=paper_id,
                skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
                pipeline_version=pipeline_version,
                cancellation_callback=cancellation_callback,
            )
        workspace = self._resolve_pdf_workspace(
            session_id, paper_id=paper_id, before_workspace_ids=before_workspace_ids
        )
        if workspace is not None and self._analysis_succeeded(result):
            artifact_repository.add_reference(
                artifact.id, ref_kind="paper_workspace", ref_id=workspace.id,
                metadata={"paper_id": workspace.paper_id},
            )
        return result

    def _store_pdf_blob(self, session_id: str, content: bytes, *, source: str) -> BlobArtifact:
        blob_store, artifact_repository = self._blob_dependencies()
        blob_store.ensure_bucket()
        stored = blob_store.put(content, kind="pdf", content_type="application/pdf")
        artifact = artifact_repository.upsert_artifact(stored, retention_policy="durable")
        artifact_repository.add_reference(
            artifact.id, ref_kind="session", ref_id=session_id, metadata={"source": source}
        )
        return artifact

    def _new_pdf_upload(
        self,
        session_id: str,
        *,
        expected_sha256: str,
        size_bytes: int,
        content_type: str,
        expires_seconds: int,
    ) -> PdfUpload:
        blob_store, _, _ = self._pdf_upload_dependencies()
        if content_type != "application/pdf":
            raise PdfUploadInvalidContentError("PDF upload content type must be application/pdf.")
        if size_bytes <= 0 or size_bytes > MAX_LOCAL_PDF_BYTES:
            raise PdfUploadSizeMismatchError(
                f"PDF upload size must be between 1 and {MAX_LOCAL_PDF_BYTES} bytes."
            )
        if not (MIN_UPLOAD_URL_EXPIRES_SECONDS <= expires_seconds <= MAX_UPLOAD_URL_EXPIRES_SECONDS):
            raise ValueError(
                f"expires_seconds must be between {MIN_UPLOAD_URL_EXPIRES_SECONDS} "
                f"and {MAX_UPLOAD_URL_EXPIRES_SECONDS}."
            )
        upload_id = str(uuid4())
        blob_store.ensure_bucket()
        return PdfUpload(
            id=upload_id, session_id=session_id,
            object_key=f"uploads/{session_id}/{upload_id}.pdf",
            expected_sha256=expected_sha256, size_bytes=size_bytes,
            content_type=content_type,
            expires_at=utc_now() + timedelta(seconds=expires_seconds),
        )

    def _validate_pdf_upload_metadata(self, upload: PdfUpload, metadata) -> None:
        if metadata.content_type != "application/pdf":
            raise PdfUploadInvalidContentError(
                f"PDF upload content type mismatch: got {metadata.content_type!r}."
            )
        if metadata.size_bytes <= 0 or metadata.size_bytes > MAX_LOCAL_PDF_BYTES:
            raise PdfUploadSizeMismatchError(
                f"PDF upload size must be between 1 and {MAX_LOCAL_PDF_BYTES} bytes; "
                f"got {metadata.size_bytes}."
            )
        if upload.size_bytes != metadata.size_bytes:
            raise PdfUploadSizeMismatchError(
                f"PDF upload size mismatch: expected {upload.size_bytes}, "
                f"got {metadata.size_bytes}."
            )

    def _validate_pdf_upload_content(self, upload: PdfUpload, content: bytes) -> None:
        if not content.startswith(b"%PDF-"):
            raise PdfUploadInvalidContentError("PDF upload must start with %PDF- magic bytes.")
        if upload.size_bytes is not None and len(content) != upload.size_bytes:
            raise PdfUploadSizeMismatchError(
                f"PDF upload size mismatch: expected {upload.size_bytes}, got {len(content)}."
            )
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if upload.expected_sha256 != actual_sha256:
            raise PdfUploadChecksumMismatchError(
                f"PDF upload checksum mismatch: expected {upload.expected_sha256}, got {actual_sha256}."
            )

    @staticmethod
    def _analysis_succeeded(result: HandlerResult) -> bool:
        return result.phase == "qa" and not result.needs_analysis and not result.errors and result.error is None

    def _blob_dependencies(self) -> tuple[BlobStore, BlobArtifactRepository]:
        if self.blob_store is None or self.blob_artifact_repository is None:
            raise BlobStorageNotConfiguredError()
        return self.blob_store, self.blob_artifact_repository

    def _pdf_upload_dependencies(
        self,
    ) -> tuple[BlobStore, BlobArtifactRepository, PdfUploadRepository]:
        blob_store, artifact_repository = self._blob_dependencies()
        if self.pdf_upload_repository is None:
            raise PdfUploadNotConfiguredError()
        return blob_store, artifact_repository, self.pdf_upload_repository

    def _pdf_upload_repository(self) -> PdfUploadRepository:
        if self.pdf_upload_repository is None:
            raise PdfUploadNotConfiguredError()
        return self.pdf_upload_repository

    def _analyze_pdf_path(
        self,
        session_id: str,
        pdf_path: str,
        *,
        user_content: str,
        paper_id: str | None,
        skip_arxiv_metadata_fetch: bool,
        pipeline_version: str = "v1",
        cancellation_callback=None,
    ) -> HandlerResult:
        kwargs = {
            "input_type": "pdf",
            "input_value": pdf_path,
            "user_content": user_content,
            "expected_paper_id": paper_id,
            "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
            "pipeline_version": pipeline_version,
        }
        if cancellation_callback is not None:
            kwargs["cancellation_callback"] = cancellation_callback
        return self.handler.analyze_paper_input(session_id, **kwargs)

    def _workspace_ids(self, session_id: str) -> set[str]:
        if self.artifact_repository is None:
            return set()
        return {
            workspace.id
            for workspace in self.artifact_repository.list_workspaces(session_id)
        }

    def _resolve_pdf_workspace(
        self,
        session_id: str,
        *,
        paper_id: str | None,
        before_workspace_ids: set[str],
    ) -> PaperWorkspace | None:
        if self.artifact_repository is None:
            return None
        if paper_id is not None:
            return self.artifact_repository.get_workspace(session_id, paper_id)
        new_workspaces = [
            workspace
            for workspace in self.artifact_repository.list_workspaces(session_id)
            if workspace.id not in before_workspace_ids
        ]
        return new_workspaces[0] if len(new_workspaces) == 1 else None

    def _hydrate_reusable_workspace(
        self,
        session_id: str,
        source_workspace: PaperWorkspace,
        *,
        cache_reason: str,
    ) -> HandlerResult:
        _ = cache_reason
        self.handler.store.require_session(session_id)
        artifact_repository, chunk_repository, retrieval_layer = (
            self._paper_cache_dependencies()
        )
        cloned_workspace = artifact_repository.clone_workspace(
            source_workspace_id=source_workspace.id,
            target_session_id=session_id,
        )
        cloned_chunks = chunk_repository.clone_for_session(
            source_session_id=source_workspace.session_id,
            target_session_id=session_id,
            paper_id=source_workspace.paper_id,
        )
        if not cloned_chunks:
            raise PaperCacheHydrationEmptyChunksError(
                session_id=session_id,
                paper_id=source_workspace.paper_id,
        )
        retrieval_layer.upsert_chunks(cloned_chunks)
        self.handler.store.add_active_paper(session_id, cloned_workspace.paper_id)
        self.handler.store.update_phase(session_id, "qa")
        return HandlerResult(
            session_id=session_id,
            response_text=(
                cloned_workspace.full_markdown_report
                or f"Loaded analysis for {cloned_workspace.paper_id}."
            ),
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=[cloned_workspace.paper_id],
            artifact_refs=[
                f"paper_workspace:{cloned_workspace.id}",
            ],
            user_turn_id="reused-analysis-user-turn",
            assistant_turn_id="reused-analysis-assistant-turn",
        )

    def _paper_cache_dependencies(self):
        retrieval_layer = getattr(self.handler, "retrieval_layer", None)
        if (
            self.artifact_repository is None
            or self.paper_chunk_repository is None
            or retrieval_layer is None
        ):
            raise PaperCacheHydrationNotConfiguredError()
        return self.artifact_repository, self.paper_chunk_repository, retrieval_layer

    def ask_question(self, session_id: str, question: str) -> HandlerResult:
        return self.handler.handle_message(session_id, question)

    def synthesize_papers(
        self,
        session_id: str,
        prompt: str | None = None,
        paper_ids: list[str] | None = None,
    ) -> SynthesisAgentResult:
        session = self.handler.store.require_session(session_id)
        requested_ids = (
            list(paper_ids)
            if paper_ids is not None
            else list(session.active_paper_ids)
        )
        if not requested_ids:
            raise NoActivePapersError(session_id)
        workspaces = self._load_request_workspaces(session_id, requested_ids)
        comparison = self._latest_relevant_comparison(
            session_id,
            [workspace.paper_id for workspace in workspaces],
        )
        return synthesize_workspaces(
            session_id=session_id,
            persona=session.persona,
            workspaces=workspaces,
            prompt=prompt,
            comparison=comparison,
            config={
                "configurable": {
                    "session_id": session_id,
                    "agent_run_persistence": self.handler.agent_run_persistence,
                }
            },
        )

    def discover_papers(self, session_id: str, topic_message: str) -> HandlerResult:
        topic_message = topic_message.strip()
        if not _looks_like_discovery_message(topic_message):
            topic_message = f"Find papers about {topic_message}"
        return self.handler.handle_message(session_id, topic_message)

    def select_papers(self, session_id: str, selection_message: str) -> HandlerResult:
        session = self.handler.store.require_session(session_id)
        if session.phase != "selection":
            raise InvalidSessionPhaseError(expected="selection", actual=session.phase)
        return self.handler.handle_message(session_id, selection_message)

    def analyze_selected_papers(self, session_id: str) -> HandlerResult:
        if self.selected_candidate_resolver is None:
            raise RuntimeError("Selected candidate analysis is not configured.")
        if self.candidate_repository is None:
            raise RuntimeError("Search candidate repository is not configured.")

        selected = self.selected_candidate_resolver.resolve(session_id)
        result = self.handler.analyze_selected_papers(session_id, selected.urls)
        if (
            result.intent == "analyze_paper"
            and result.phase == "qa"
            and not result.needs_analysis
        ):
            for candidate_id in selected.candidate_ids:
                self.candidate_repository.update_status(candidate_id, "analyzed")
        return result

    def get_session(self, session_id: str) -> Session:
        return self.handler.store.require_session(session_id)

    def list_turns(self, session_id: str, *, limit: int = 50) -> list[Turn]:
        self.handler.store.require_session(session_id)
        return self.handler.store.list_recent_turns(session_id, limit=limit)

    def list_paper_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        self.handler.store.require_session(session_id)
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")
        return self.artifact_repository.list_workspaces(session_id)

    def get_paper_workspace(self, session_id: str, paper_id: str) -> PaperWorkspace:
        self.handler.store.require_session(session_id)
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")
        workspace = self.artifact_repository.get_workspace(session_id, paper_id)
        if workspace is None:
            raise PaperWorkspaceNotFoundError(
                session_id=session_id,
                paper_id=paper_id,
            )
        return workspace

    def get_latest_comparison(self, session_id: str) -> ComparisonArtifact:
        self.handler.store.require_session(session_id)
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")
        comparison = self.artifact_repository.latest_comparison(session_id)
        if comparison is None:
            raise ComparisonNotFoundError(session_id)
        return comparison

    def compare_papers(
        self,
        session_id: str,
        paper_ids: list[str] | None = None,
        prompt: str | None = None,
    ) -> ComparisonArtifact:
        session = self.handler.store.require_session(session_id)
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")

        requested_ids = (
            list(paper_ids)
            if paper_ids is not None
            else list(session.active_paper_ids)
        )
        workspaces = self._load_request_workspaces(session_id, requested_ids)
        requested_ids = [workspace.paper_id for workspace in workspaces]
        result = compare_workspaces(
            session_id=session_id,
            workspaces=workspaces,
            prompt=prompt,
            config={
                "configurable": {
                    "session_id": session_id,
                    "agent_run_persistence": self.handler.agent_run_persistence,
                }
            },
        )
        artifact = ComparisonArtifact(
            session_id=session_id,
            paper_ids=requested_ids,
            comparison_report_json=result.report.model_dump(mode="json"),
            comparison_markdown=result.markdown,
        )
        return self.artifact_repository.save_comparison(artifact)

    def _workflow_jobs(self) -> WorkflowJobRepository:
        if self.workflow_job_repository is None:
            raise WorkflowJobNotConfiguredError()
        return self.workflow_job_repository

    def _load_request_workspaces(
        self,
        session_id: str,
        requested_ids: list[str],
    ) -> list[PaperWorkspace]:
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")

        requested_ids = list(dict.fromkeys(requested_ids))
        if len(requested_ids) < 2:
            raise NotEnoughPapersForComparisonError(
                session_id=session_id,
                paper_ids=requested_ids,
            )

        workspaces_by_id = {
            workspace.paper_id: workspace
            for workspace in self.artifact_repository.list_workspaces(session_id)
        }
        missing = [
            paper_id
            for paper_id in requested_ids
            if paper_id not in workspaces_by_id
        ]
        if missing:
            raise PaperWorkspaceNotFoundError(
                session_id=session_id,
                paper_id=missing[0],
            )

        workspaces = [workspaces_by_id[paper_id] for paper_id in requested_ids]
        for workspace in workspaces:
            if workspace.pipeline_stage in _FAILED_WORKSPACE_STAGES:
                raise PaperWorkspaceNotReadyError(
                    session_id=session_id,
                    paper_id=workspace.paper_id,
                    pipeline_stage=workspace.pipeline_stage,
                )
        return workspaces

    def _latest_relevant_comparison(
        self,
        session_id: str,
        paper_ids: list[str],
    ) -> ComparisonArtifact | None:
        if self.artifact_repository is None:
            raise RuntimeError("Paper workspace repository is not configured.")
        comparison = self.artifact_repository.latest_comparison(session_id)
        if comparison is None:
            return None
        selected = set(paper_ids)
        if not selected.intersection(comparison.paper_ids):
            return None
        return comparison

    def health(self) -> HealthStatus:
        if self.health_checker is None:
            return HealthStatus(healthy=True, checks={"basic": "ok"})
        return self.health_checker.check()


def _looks_like_discovery_message(message: str) -> bool:
    normalized = message.casefold()
    discovery_words = ("find", "search", "discover", "recommend")
    target_words = ("paper", "papers", "literature", "research")
    return any(word in normalized for word in discovery_words) and any(
        word in normalized for word in target_words
    )
