import hashlib
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.in_memory_session_store import SessionNotFoundError
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_artifacts import BlobArtifact, BlobReference
from models.blob_storage import BlobObjectMetadata, StoredBlobObject
from models.discovery import SearchCandidate
from models.api import HealthStatus
from models.errors import ErrorCodes, make_error
from models.jobs import WorkflowJob
from models.pdf_uploads import PdfUpload
from models.session import utc_now
from models.session import HandlerResult, Session, Turn
from services.blob_store import BlobNotFoundError, BlobStoreUnavailableError
from services.paperintel_service import (
    BlobStorageNotConfiguredError,
    ComparisonNotFoundError,
    InvalidPdfInputError,
    InvalidSessionPhaseError,
    NoActivePapersError,
    NotEnoughPapersForComparisonError,
    PaperIntelService,
    PaperWorkspaceNotFoundError,
    InvalidWorkflowJobInputError,
    PdfUploadChecksumMismatchError,
    PdfUploadSizeMismatchError,
    WorkflowJobNotFoundError,
)
from services.selected_candidate_resolver import NoSelectedCandidatesError


class FakeHandler:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.created_sessions = []
        self.messages = []
        self.analysis_input_calls = []
        self.selected_analysis_calls = []
        self.selected_analysis_result = None
        self.agent_run_persistence = InMemoryAgentRunPersistence()

    def create_session(self, *, persona="engineer", original_query=None):
        session = self.store.create_session(
            persona=persona,
            original_query=original_query,
        )
        self.created_sessions.append(session)
        return session

    def handle_message(self, session_id, message):
        self.messages.append((session_id, message))
        return HandlerResult(
            session_id=session_id,
            response_text=f"handled: {message}",
            phase="qa",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_selected_papers(self, session_id, urls):
        self.selected_analysis_calls.append((session_id, list(urls)))
        return self.selected_analysis_result or HandlerResult(
            session_id=session_id,
            response_text="selected analysis complete",
            phase="qa",
            intent="analyze_paper",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_paper_input(
        self,
        session_id,
        *,
        input_type,
        input_value,
        user_content=None,
        expected_paper_id=None,
        skip_arxiv_metadata_fetch=False,
    ):
        self.analysis_input_calls.append(
            {
                "session_id": session_id,
                "input_type": input_type,
                "input_value": input_value,
                "user_content": user_content,
                "expected_paper_id": expected_paper_id,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
            }
        )
        return HandlerResult(
            session_id=session_id,
            response_text="pdf analysis complete",
            phase="qa",
            intent="analyze_paper",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )


class FakeStore:
    def __init__(self) -> None:
        self.sessions = {}
        self.turns = {}

    def create_session(self, *, persona="engineer", original_query=None):
        session = Session(persona=persona, original_query=original_query)
        self.sessions[session.id] = session
        self.turns[session.id] = []
        return session

    def require_session(self, session_id):
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return self.sessions[session_id]

    def list_recent_turns(self, session_id, limit=20):
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return self.turns[session_id][-limit:]


class FakeHealthChecker:
    def __init__(self, status=None) -> None:
        self.status = status or HealthStatus(
            healthy=True,
            checks={"postgres": "ok", "qdrant": "ok"},
        )
        self.calls = 0

    def check(self):
        self.calls += 1
        return self.status


class FakeSelectedCandidateResolver:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = []

    def resolve(self, session_id):
        self.calls.append(session_id)

        class Selected:
            def __init__(self, candidates):
                self.candidates = candidates

            @property
            def urls(self):
                return [candidate.url for candidate in self.candidates]

            @property
            def candidate_ids(self):
                return [candidate.id for candidate in self.candidates]

        return Selected(self.candidates)


class FailingSelectedCandidateResolver:
    def resolve(self, session_id):
        raise NoSelectedCandidatesError(session_id)


class FakeCandidateRepository:
    def __init__(self):
        self.updates = []

    def update_status(self, candidate_id, status):
        self.updates.append((candidate_id, status))
        return None


class FakeArtifactRepository:
    def __init__(self, *, session_id: str = "session-1") -> None:
        self.workspaces = [
            PaperWorkspace(
                session_id=session_id,
                paper_id="1706.03762",
                title="Attention Is All You Need",
                source_url="https://arxiv.org/abs/1706.03762",
                pipeline_stage="completed",
                full_markdown_report="# Report",
            )
        ]
        self.comparison = ComparisonArtifact(
            session_id=session_id,
            paper_ids=["1706.03762", "2401.00001"],
            comparison_markdown="# Comparison",
        )

    def list_workspaces(self, session_id):
        return [
            workspace
            for workspace in self.workspaces
            if workspace.session_id == session_id
        ]

    def get_workspace(self, session_id, paper_id):
        for workspace in self.workspaces:
            if workspace.session_id == session_id and workspace.paper_id == paper_id:
                return workspace
        return None

    def latest_comparison(self, session_id):
        if self.comparison.session_id == session_id:
            return self.comparison
        return None

    def save_comparison(self, artifact):
        self.comparison = artifact
        return artifact


class FakeBlobStore:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.put_calls = []
        self.materialized_paths = []
        self.deleted_materialized_paths = []
        self.objects = {}
        self.deleted_objects = []
        self.head_calls = []
        self.presigned_calls = []

    def ensure_bucket(self):
        self.ensure_calls += 1

    def put(self, content, *, kind, content_type=None):
        self.put_calls.append(
            {"content": content, "kind": kind, "content_type": content_type}
        )
        content_hash = hashlib.sha256(content).hexdigest()
        object_key = f"papers/sha256/{content_hash[:2]}/{content_hash}.pdf"
        self.objects[object_key] = {"content": content, "content_type": content_type or "application/pdf"}
        return StoredBlobObject(
            kind=kind, object_key=object_key, bucket_name="paperintel-test",
            content_hash=content_hash, content_type=content_type or "application/pdf",
            size_bytes=len(content),
        )

    def put_staging(self, object_key, content, *, content_type):
        self.objects[object_key] = {"content": content, "content_type": content_type}

    def create_presigned_put(self, object_key, *, content_type, expires_seconds):
        self.presigned_calls.append((object_key, content_type, expires_seconds))
        return f"https://uploads.example/{object_key}"

    def delete(self, object_key):
        self.deleted_objects.append(object_key)
        self.objects.pop(object_key, None)

    def head_object(self, object_key):
        self.head_calls.append(object_key)
        if object_key not in self.objects:
            raise BlobNotFoundError(f"Blob object not found: {object_key}")
        stored = self.objects[object_key]
        return BlobObjectMetadata(
            object_key=object_key, content_type=stored["content_type"],
            size_bytes=len(stored["content"]),
        )

    @contextmanager
    def materialize(self, object_key, *, expected_sha256=None, max_bytes=None):
        if object_key not in self.objects:
            raise BlobNotFoundError(f"Blob object not found: {object_key}")
        if max_bytes is not None and len(self.objects[object_key]["content"]) > max_bytes:
            from services.blob_store import BlobSizeLimitError
            raise BlobSizeLimitError("object exceeds limit")
        with NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="paperintel_test_blob_",
            delete=False,
        ) as temp_file:
            temp_file.write(self.objects[object_key]["content"])
            temp_path = temp_file.name
        self.materialized_paths.append(temp_path)
        try:
            yield temp_path
        finally:
            Path(temp_path).unlink(missing_ok=True)
            self.deleted_materialized_paths.append(temp_path)


class FakeBlobArtifactRepository:
    def __init__(self) -> None:
        self.artifacts = {}
        self.references = []
        self.accessed = []

    def upsert_artifact(self, stored, **kwargs):
        key = (stored.kind, stored.content_hash)
        if key not in self.artifacts:
            self.artifacts[key] = BlobArtifact(
                kind=stored.kind,
                object_key=stored.object_key,
                bucket_name=stored.bucket_name,
                content_hash=stored.content_hash,
                content_type=stored.content_type,
                size_bytes=stored.size_bytes,
                storage_backend=stored.storage_backend,
                retention_policy=kwargs.get("retention_policy", "durable"),
            )
        return self.artifacts[key]

    def add_reference(self, blob_id, *, ref_kind, ref_id, metadata=None):
        existing = next(
            (
                reference
                for reference in self.references
                if (reference.blob_id, reference.ref_kind, reference.ref_id)
                == (blob_id, ref_kind, ref_id)
            ),
            None,
        )
        if existing is not None:
            return existing
        reference = BlobReference(
            blob_id=blob_id,
            ref_kind=ref_kind,
            ref_id=ref_id,
            metadata=metadata or {},
        )
        self.references.append(reference)
        return reference

    def mark_accessed(self, blob_id):
        self.accessed.append(blob_id)
        artifact = next(
            artifact for artifact in self.artifacts.values() if artifact.id == blob_id
        )
        return artifact

    def get_artifact(self, blob_id):
        return next(
            (artifact for artifact in self.artifacts.values() if artifact.id == blob_id),
            None,
        )

    def has_active_reference(self, blob_id, *, ref_kind, ref_id):
        return any(
            reference.blob_id == blob_id
            and reference.ref_kind == ref_kind
            and reference.ref_id == ref_id
            and reference.status == "active"
            for reference in self.references
        )


class FakePdfUploadRepository:
    def __init__(self) -> None:
        self.uploads = {}

    def create(self, upload):
        self.uploads[upload.id] = upload
        return upload

    def get(self, upload_id):
        return self.uploads.get(upload_id)

    def mark_uploaded(self, upload_id):
        upload = self.uploads[upload_id].model_copy(update={"status": "uploaded"})
        self.uploads[upload_id] = upload
        return upload

    def finalize(self, upload_id, *, blob_id, actual_sha256, size_bytes):
        upload = self.uploads[upload_id].model_copy(
            update={
                "status": "finalized",
                "blob_id": blob_id,
                "actual_sha256": actual_sha256,
                "size_bytes": size_bytes,
                "finalized_at": utc_now(),
                "error_json": None,
            }
        )
        self.uploads[upload_id] = upload
        return upload

    def mark_failed(self, upload_id, *, error_json):
        upload = self.uploads[upload_id].model_copy(
            update={"status": "failed", "error_json": error_json}
        )
        self.uploads[upload_id] = upload
        return upload


class FakeWorkflowJobRepository:
    def __init__(self) -> None:
        self.jobs = {}
        self.created = []
        self.canceled = []

    def create(self, job):
        self.jobs[job.id] = job
        self.created.append(job)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def list_for_session(self, session_id, limit=50):
        return [
            job
            for job in self.created
            if job.session_id == session_id
        ][:limit]

    def mark_canceled(self, job_id):
        job = self.jobs[job_id].model_copy(update={"status": "canceled"})
        self.jobs[job_id] = job
        self.canceled.append(job_id)
        return job


def _candidate(candidate_id: str) -> SearchCandidate:
    return SearchCandidate(
        id=candidate_id,
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        status="selected",
        title=f"Paper {candidate_id}",
        url=f"https://arxiv.org/abs/{candidate_id}",
        arxiv_id=candidate_id,
    )


def test_service_create_session_delegates_to_handler_with_persona():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)

    session = service.create_session(
        persona="researcher",
        original_query="agent memory",
    )

    assert session.persona == "researcher"
    assert session.original_query == "agent memory"
    assert handler.created_sessions == [session]


def test_service_handle_message_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.handle_message(session.id, "hello")

    assert result.response_text == "handled: hello"
    assert handler.messages == [(session.id, "hello")]


def test_service_analyze_paper_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.analyze_paper(session.id, "https://arxiv.org/abs/1706.03762")

    assert result.response_text == "handled: https://arxiv.org/abs/1706.03762"
    assert handler.messages == [(session.id, "https://arxiv.org/abs/1706.03762")]


def test_service_enqueue_analyze_paper_creates_queued_job():
    repository = FakeWorkflowJobRepository()
    service = PaperIntelService(
        handler=FakeHandler(),
        workflow_job_repository=repository,
    )
    session = service.create_session()

    job = service.enqueue_analyze_paper(
        session.id,
        " https://arxiv.org/abs/1706.03762 ",
    )

    assert job.status == "queued"
    assert job.kind == "analyze_paper"
    assert job.session_id == session.id
    assert job.input_json == {"paper_url": "https://arxiv.org/abs/1706.03762"}
    assert repository.created == [job]


def test_service_enqueue_analyze_paper_requires_non_empty_url():
    service = PaperIntelService(
        handler=FakeHandler(),
        workflow_job_repository=FakeWorkflowJobRepository(),
    )
    session = service.create_session()

    with pytest.raises(InvalidWorkflowJobInputError):
        service.enqueue_analyze_paper(session.id, "   ")


def test_service_enqueue_analyze_selected_creates_queued_job():
    repository = FakeWorkflowJobRepository()
    service = PaperIntelService(
        handler=FakeHandler(),
        workflow_job_repository=repository,
    )
    session = service.create_session()

    job = service.enqueue_analyze_selected(session.id)

    assert job.status == "queued"
    assert job.kind == "analyze_selected"
    assert job.session_id == session.id
    assert job.input_json == {}


def test_service_get_list_and_cancel_workflow_jobs():
    repository = FakeWorkflowJobRepository()
    service = PaperIntelService(
        handler=FakeHandler(),
        workflow_job_repository=repository,
    )
    session = service.create_session()
    job = service.enqueue_analyze_selected(session.id)

    assert service.get_workflow_job(job.id) == job
    assert service.list_workflow_jobs(session.id) == [job]

    canceled = service.cancel_workflow_job(job.id)

    assert canceled.status == "canceled"
    assert repository.canceled == [job.id]


def test_service_get_workflow_job_raises_when_missing():
    service = PaperIntelService(
        handler=FakeHandler(),
        workflow_job_repository=FakeWorkflowJobRepository(),
    )

    with pytest.raises(WorkflowJobNotFoundError):
        service.get_workflow_job("missing")


def test_service_analyze_pdf_passes_expected_paper_id_to_handler(tmp_path):
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    pdf_path = tmp_path / "2501.12948.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    result = service.analyze_pdf(
        session.id,
        str(pdf_path),
        paper_id="2501.12948",
        skip_arxiv_metadata_fetch=True,
    )

    assert result.response_text == "pdf analysis complete"
    assert handler.analysis_input_calls == [
        {
            "session_id": session.id,
            "input_type": "pdf",
            "input_value": str(pdf_path),
            "user_content": "Analyze local PDF 2501.12948",
            "expected_paper_id": "2501.12948",
            "skip_arxiv_metadata_fetch": True,
        }
    ]


def test_service_analyze_pdf_uses_durable_blob_materialization(tmp_path):
    handler = FakeHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    artifact_repository = FakeArtifactRepository()
    session = handler.create_session()
    workspace = PaperWorkspace(
        session_id=session.id,
        paper_id="2501.12948",
        source_url="local:2501.12948",
        pipeline_stage="completed",
    )
    artifact_repository.workspaces.append(workspace)
    service = PaperIntelService(
        handler=handler,
        artifact_repository=artifact_repository,
        blob_store=blob_store,
        blob_artifact_repository=blob_repository,
    )
    pdf_path = tmp_path / "2501.12948.pdf"
    pdf_bytes = b"%PDF-1.7\nbody"
    pdf_path.write_bytes(pdf_bytes)

    result = service.analyze_pdf(session.id, str(pdf_path), paper_id="2501.12948")

    assert result.response_text == "pdf analysis complete"
    assert blob_store.ensure_calls == 1
    assert blob_store.put_calls == [
        {"content": pdf_bytes, "kind": "pdf", "content_type": "application/pdf"}
    ]
    assert len(blob_repository.artifacts) == 1
    blob_id = next(iter(blob_repository.artifacts.values())).id
    assert blob_repository.accessed == [blob_id]
    assert [(ref.ref_kind, ref.ref_id) for ref in blob_repository.references] == [
        ("session", session.id),
        ("paper_workspace", workspace.id),
    ]
    analyzed_path = handler.analysis_input_calls[0]["input_value"]
    assert analyzed_path == blob_store.materialized_paths[0]
    assert analyzed_path != str(pdf_path)
    assert not Path(analyzed_path).exists()
    assert blob_store.deleted_materialized_paths == [analyzed_path]


def test_service_analyze_pdf_reuses_durable_blob_and_references(tmp_path):
    handler = FakeHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    service = PaperIntelService(
        handler=handler,
        blob_store=blob_store,
        blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    service.analyze_pdf(session.id, str(pdf_path))
    service.analyze_pdf(session.id, str(pdf_path))

    assert len(blob_repository.artifacts) == 1
    assert len(blob_repository.references) == 1
    assert blob_repository.references[0].ref_kind == "session"
    assert len(blob_repository.accessed) == 2


def test_service_analyze_pdf_skips_ambiguous_workspace_reference(tmp_path):
    class AddingHandler(FakeHandler):
        def __init__(self, artifact_repository):
            super().__init__()
            self.artifact_repository = artifact_repository

        def analyze_paper_input(self, session_id, **kwargs):
            self.artifact_repository.workspaces.extend(
                [
                    PaperWorkspace(
                        session_id=session_id,
                        paper_id="paper-a",
                        source_url="local:paper-a",
                        pipeline_stage="completed",
                    ),
                    PaperWorkspace(
                        session_id=session_id,
                        paper_id="paper-b",
                        source_url="local:paper-b",
                        pipeline_stage="completed",
                    ),
                ]
            )
            return super().analyze_paper_input(session_id, **kwargs)

    artifact_repository = FakeArtifactRepository()
    handler = AddingHandler(artifact_repository)
    blob_repository = FakeBlobArtifactRepository()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=artifact_repository,
        blob_store=FakeBlobStore(),
        blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    result = service.analyze_pdf(session.id, str(pdf_path))

    assert result.response_text == "pdf analysis complete"
    assert [reference.ref_kind for reference in blob_repository.references] == [
        "session"
    ]


def test_service_analyze_pdf_links_unique_generated_workspace(tmp_path):
    class AddingHandler(FakeHandler):
        def __init__(self, artifact_repository):
            super().__init__()
            self.artifact_repository = artifact_repository
            self.workspace = None

        def analyze_paper_input(self, session_id, **kwargs):
            self.workspace = PaperWorkspace(
                session_id=session_id,
                paper_id="generated-paper",
                source_url="local:generated-paper",
                pipeline_stage="completed",
            )
            self.artifact_repository.workspaces.append(self.workspace)
            return super().analyze_paper_input(session_id, **kwargs)

    artifact_repository = FakeArtifactRepository()
    handler = AddingHandler(artifact_repository)
    blob_repository = FakeBlobArtifactRepository()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=artifact_repository,
        blob_store=FakeBlobStore(),
        blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    service.analyze_pdf(session.id, str(pdf_path))

    assert handler.workspace is not None
    assert [(reference.ref_kind, reference.ref_id) for reference in blob_repository.references] == [
        ("session", session.id),
        ("paper_workspace", handler.workspace.id),
    ]


def test_service_analyze_pdf_propagates_blob_store_outage(tmp_path):
    class UnavailableBlobStore(FakeBlobStore):
        def ensure_bucket(self):
            raise BlobStoreUnavailableError("provider down")

    service = PaperIntelService(
        handler=FakeHandler(),
        blob_store=UnavailableBlobStore(),
        blob_artifact_repository=FakeBlobArtifactRepository(),
    )
    session = service.create_session()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    with pytest.raises(BlobStoreUnavailableError, match="provider down"):
        service.analyze_pdf(session.id, str(pdf_path))


def test_service_analyze_pdf_rejects_missing_session_before_blob_side_effects(tmp_path):
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    service = PaperIntelService(
        handler=FakeHandler(),
        blob_store=blob_store,
        blob_artifact_repository=blob_repository,
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    with pytest.raises(SessionNotFoundError):
        service.analyze_pdf("missing-session", str(pdf_path))

    assert blob_store.ensure_calls == 0
    assert blob_store.put_calls == []
    assert blob_repository.artifacts == {}
    assert blob_repository.references == []


def test_service_analyze_pdf_rejects_partial_blob_storage_configuration(tmp_path):
    service = PaperIntelService(handler=FakeHandler(), blob_store=FakeBlobStore())
    session = service.create_session()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    with pytest.raises(BlobStorageNotConfiguredError):
        service.analyze_pdf(session.id, str(pdf_path))


def test_service_analyze_pdf_rejects_missing_file():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(InvalidPdfInputError, match="does not exist"):
        service.analyze_pdf(session.id, "/tmp/missing-paper.pdf")


def test_service_analyze_pdf_rejects_non_pdf_magic_bytes(tmp_path):
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()
    pdf_path = tmp_path / "not-a-pdf.pdf"
    pdf_path.write_bytes(b"hello")

    with pytest.raises(InvalidPdfInputError, match="magic bytes"):
        service.analyze_pdf(session.id, str(pdf_path))


def test_service_analyze_pdf_rejects_oversized_pdf(tmp_path, monkeypatch):
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()
    pdf_path = tmp_path / "large.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr("services.paperintel_service.MAX_LOCAL_PDF_BYTES", 4)

    with pytest.raises(InvalidPdfInputError, match="too large"):
        service.analyze_pdf(session.id, str(pdf_path))


def test_service_ask_question_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.ask_question(session.id, "What is the contribution?")

    assert result.response_text == "handled: What is the contribution?"
    assert handler.messages == [(session.id, "What is the contribution?")]


@pytest.mark.parametrize("active_ids", [["paper-1"], ["paper-1", "paper-1"]])
def test_service_synthesize_papers_requires_two_distinct_active_papers(active_ids):
    handler = FakeHandler()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(),
    )
    session = service.create_session()
    handler.store.sessions[session.id] = session.model_copy(
        update={"active_paper_ids": active_ids}
    )

    with pytest.raises(NotEnoughPapersForComparisonError):
        service.synthesize_papers(session.id, "Compare deployment risks.")


def test_service_synthesize_papers_requires_active_papers():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(NoActivePapersError):
        service.synthesize_papers(session.id)


def test_service_discover_papers_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.discover_papers(session.id, "Find papers about agent memory")

    assert result.response_text == "handled: Find papers about agent memory"
    assert handler.messages == [(session.id, "Find papers about agent memory")]


def test_service_discover_papers_wraps_bare_topic_for_routing():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    result = service.discover_papers(session.id, "agent memory")

    assert result.response_text == "handled: Find papers about agent memory"
    assert handler.messages == [(session.id, "Find papers about agent memory")]


def test_service_select_papers_delegates_to_handler():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    handler.store.sessions[session.id] = session.model_copy(update={"phase": "selection"})

    result = service.select_papers(session.id, "use 1 and 3")

    assert result.response_text == "handled: use 1 and 3"
    assert handler.messages == [(session.id, "use 1 and 3")]


def test_service_select_papers_requires_selection_phase():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(InvalidSessionPhaseError):
        service.select_papers(session.id, "use 1")


def test_service_analyze_selected_papers_resolves_and_updates_statuses():
    handler = FakeHandler()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver(
            [_candidate("2401.1"), _candidate("2401.2")]
        ),
        candidate_repository=FakeCandidateRepository(),
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.response_text == "selected analysis complete"
    assert handler.selected_analysis_calls == [
        (
            session.id,
            ["https://arxiv.org/abs/2401.1", "https://arxiv.org/abs/2401.2"],
        )
    ]
    assert service.candidate_repository.updates == [
        ("2401.1", "analyzed"),
        ("2401.2", "analyzed"),
    ]


def test_service_analyze_selected_papers_updates_statuses_with_warnings():
    handler = FakeHandler()
    handler.selected_analysis_result = HandlerResult(
        session_id="session-1",
        response_text="selected analysis complete",
        phase="qa",
        intent="analyze_paper",
        errors=[
            make_error(
                ErrorCodes.WARNING,
                "benchmarks missing",
                severity="warning",
                recoverable=True,
            )
        ],
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )
    repository = FakeCandidateRepository()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver([_candidate("2401.1")]),
        candidate_repository=repository,
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.errors
    assert repository.updates == [("2401.1", "analyzed")]


def test_service_analyze_selected_papers_does_not_update_status_when_analysis_missing():
    handler = FakeHandler()
    handler.selected_analysis_result = HandlerResult(
        session_id="session-1",
        response_text="analysis missing",
        phase="selection",
        intent="analyze_paper",
        needs_analysis=True,
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )
    repository = FakeCandidateRepository()
    service = PaperIntelService(
        handler=handler,
        selected_candidate_resolver=FakeSelectedCandidateResolver([_candidate("2401.1")]),
        candidate_repository=repository,
    )
    session = service.create_session()

    result = service.analyze_selected_papers(session.id)

    assert result.needs_analysis is True
    assert repository.updates == []


def test_service_analyze_selected_papers_requires_resolver():
    service = PaperIntelService(handler=FakeHandler())
    session = service.create_session()

    with pytest.raises(RuntimeError):
        service.analyze_selected_papers(session.id)


def test_service_analyze_selected_papers_propagates_resolver_error():
    service = PaperIntelService(
        handler=FakeHandler(),
        selected_candidate_resolver=FailingSelectedCandidateResolver(),
        candidate_repository=FakeCandidateRepository(),
    )
    session = service.create_session()

    with pytest.raises(NoSelectedCandidatesError):
        service.analyze_selected_papers(session.id)


def test_service_get_session_returns_session_from_store():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()

    assert service.get_session(session.id) == session


def test_service_get_session_raises_for_missing_session():
    service = PaperIntelService(handler=FakeHandler())

    with pytest.raises(SessionNotFoundError):
        service.get_session("missing")


def test_service_list_turns_returns_history_from_store():
    handler = FakeHandler()
    service = PaperIntelService(handler=handler)
    session = service.create_session()
    first = Turn(session_id=session.id, role="user", content="first")
    second = Turn(session_id=session.id, role="assistant", content="second")
    handler.store.turns[session.id] = [first, second]

    assert service.list_turns(session.id, limit=1) == [second]


def test_service_list_turns_requires_session_before_listing():
    service = PaperIntelService(handler=FakeHandler())

    with pytest.raises(SessionNotFoundError):
        service.list_turns("missing")


def test_service_list_paper_workspaces_returns_repository_results():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    workspaces = service.list_paper_workspaces(session.id)

    assert [workspace.paper_id for workspace in workspaces] == ["1706.03762"]


def test_service_get_paper_workspace_returns_workspace():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    workspace = service.get_paper_workspace(session.id, "1706.03762")

    assert workspace.title == "Attention Is All You Need"


def test_service_get_paper_workspace_raises_404_domain_error():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    with pytest.raises(PaperWorkspaceNotFoundError):
        service.get_paper_workspace(session.id, "missing")


def test_service_get_latest_comparison_returns_artifact():
    handler = FakeHandler()
    session = handler.create_session()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(session_id=session.id),
    )

    comparison = service.get_latest_comparison(session.id)

    assert comparison.paper_ids == ["1706.03762", "2401.00001"]


def test_service_get_latest_comparison_raises_when_missing():
    handler = FakeHandler()
    service = PaperIntelService(
        handler=handler,
        artifact_repository=FakeArtifactRepository(),
    )
    session = service.create_session()

    with pytest.raises(ComparisonNotFoundError):
        service.get_latest_comparison(session.id)


def test_service_health_without_checker_returns_basic_ok():
    service = PaperIntelService(handler=FakeHandler())

    status = service.health()

    assert status.healthy is True
    assert status.checks == {"basic": "ok"}


def test_service_health_uses_checker_when_configured():
    checker = FakeHealthChecker()
    service = PaperIntelService(handler=FakeHandler(), health_checker=checker)

    status = service.health()

    assert status.healthy is True
    assert status.checks["postgres"] == "ok"
    assert checker.calls == 1


def _upload_service():
    handler = FakeHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    upload_repository = FakePdfUploadRepository()
    service = PaperIntelService(
        handler=handler,
        blob_store=blob_store,
        blob_artifact_repository=blob_repository,
        pdf_upload_repository=upload_repository,
    )
    return service, handler, blob_store, blob_repository, upload_repository


def test_service_initiate_pdf_upload_creates_presigned_contract():
    service, _, blob_store, _, upload_repository = _upload_service()
    session = service.create_session()
    digest = "a" * 64

    initiation = service.initiate_pdf_upload(
        session.id, expected_sha256=digest, size_bytes=128
    )

    assert initiation.upload.status == "initiated"
    assert initiation.upload.expected_sha256 == digest
    assert initiation.upload.object_key.startswith(f"uploads/{session.id}/")
    assert initiation.upload_url.endswith(initiation.upload.object_key)
    assert initiation.upload_headers == {"Content-Type": "application/pdf"}
    assert upload_repository.get(initiation.upload.id) == initiation.upload
    assert blob_store.presigned_calls == [
        (initiation.upload.object_key, "application/pdf", 900)
    ]


def test_service_store_pdf_upload_finalizes_durable_blob_and_cleans_staging():
    service, _, blob_store, blob_repository, _ = _upload_service()
    session = service.create_session()
    content = b"%PDF-1.7\nasync upload"

    upload = service.store_pdf_upload(session.id, content)

    assert upload.status == "finalized"
    assert upload.actual_sha256 == hashlib.sha256(content).hexdigest()
    assert upload.blob_id is not None
    assert upload.object_key in blob_store.deleted_objects
    assert upload.object_key not in blob_store.objects
    assert blob_store.presigned_calls == []
    assert [(ref.ref_kind, ref.ref_id) for ref in blob_repository.references] == [
        ("session", session.id)
    ]


def test_service_finalize_pdf_upload_marks_checksum_failure_and_deletes_staging():
    service, _, blob_store, _, upload_repository = _upload_service()
    session = service.create_session()
    initiation = service.initiate_pdf_upload(
        session.id, expected_sha256="a" * 64, size_bytes=len(b"%PDF-1.7\nbad")
    )
    blob_store.put_staging(
        initiation.upload.object_key, b"%PDF-1.7\nbad", content_type="application/pdf"
    )

    with pytest.raises(PdfUploadChecksumMismatchError):
        service.finalize_pdf_upload(session.id, initiation.upload.id)

    failed = upload_repository.get(initiation.upload.id)
    assert failed.status == "failed"
    assert failed.error_json["code"] == "PdfUploadChecksumMismatchError"
    assert initiation.upload.object_key not in blob_store.objects


def test_service_analyze_registered_pdf_blob_does_not_upload_again():
    handler = FakeHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    artifact_repository = FakeArtifactRepository()
    service = PaperIntelService(
        handler=handler, artifact_repository=artifact_repository,
        blob_store=blob_store, blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    stored = blob_store.put(b"%PDF-1.7\nregistered", kind="pdf")
    artifact = blob_repository.upsert_artifact(stored)
    blob_repository.add_reference(artifact.id, ref_kind="session", ref_id=session.id)
    blob_store.put_calls.clear()

    result = service.analyze_registered_pdf_blob(
        session.id, artifact.id, paper_id="1706.03762"
    )

    assert result.response_text == "pdf analysis complete"
    assert blob_store.put_calls == []
    assert blob_repository.accessed == [artifact.id]
    assert len(blob_store.materialized_paths) == 1


def test_service_analyze_registered_pdf_blob_rejects_missing_session_reference():
    handler = FakeHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    service = PaperIntelService(
        handler=handler, blob_store=blob_store, blob_artifact_repository=blob_repository
    )
    session = service.create_session()
    stored = blob_store.put(b"%PDF-1.7\nregistered", kind="pdf")
    artifact = blob_repository.upsert_artifact(stored)
    blob_store.materialized_paths.clear()

    with pytest.raises(InvalidPdfInputError, match="Registered PDF blob was not found"):
        service.analyze_registered_pdf_blob(session.id, artifact.id)

    assert blob_store.materialized_paths == []


def test_service_finalize_pdf_upload_rejects_oversized_object_before_materialize(monkeypatch):
    service, _, blob_store, _, upload_repository = _upload_service()
    session = service.create_session()
    initiation = service.initiate_pdf_upload(
        session.id, expected_sha256="a" * 64, size_bytes=1
    )
    blob_store.objects[initiation.upload.object_key] = {
        "content": b"x", "content_type": "application/pdf"
    }
    monkeypatch.setattr("services.paperintel_service.MAX_LOCAL_PDF_BYTES", 0)

    with pytest.raises(PdfUploadSizeMismatchError):
        service.finalize_pdf_upload(session.id, initiation.upload.id)

    assert blob_store.materialized_paths == []
    assert upload_repository.get(initiation.upload.id).status == "failed"


def test_service_store_pdf_upload_marks_failed_when_staging_write_fails():
    class FailingStagingBlobStore(FakeBlobStore):
        def put_staging(self, object_key, content, *, content_type):
            raise BlobStoreUnavailableError("staging unavailable")

    handler = FakeHandler()
    upload_repository = FakePdfUploadRepository()
    service = PaperIntelService(
        handler=handler, blob_store=FailingStagingBlobStore(),
        blob_artifact_repository=FakeBlobArtifactRepository(),
        pdf_upload_repository=upload_repository,
    )
    session = service.create_session()

    with pytest.raises(BlobStoreUnavailableError, match="staging unavailable"):
        service.store_pdf_upload(session.id, b"%PDF-1.7\ncontent")

    upload = next(iter(upload_repository.uploads.values()))
    assert upload.status == "failed"
    assert upload.error_json["code"] == "BlobStoreUnavailableError"


def test_service_initiate_pdf_upload_marks_failed_when_presign_fails():
    class FailingPresignBlobStore(FakeBlobStore):
        def create_presigned_put(self, object_key, *, content_type, expires_seconds):
            raise BlobStoreUnavailableError("presign unavailable")

    handler = FakeHandler()
    upload_repository = FakePdfUploadRepository()
    service = PaperIntelService(
        handler=handler, blob_store=FailingPresignBlobStore(),
        blob_artifact_repository=FakeBlobArtifactRepository(),
        pdf_upload_repository=upload_repository,
    )
    session = service.create_session()

    with pytest.raises(BlobStoreUnavailableError, match="presign unavailable"):
        service.initiate_pdf_upload(
            session.id, expected_sha256="a" * 64, size_bytes=1
        )

    upload = next(iter(upload_repository.uploads.values()))
    assert upload.status == "failed"
    assert upload.error_json["code"] == "BlobStoreUnavailableError"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size_bytes": 0}, "between 1"),
        ({"size_bytes": 1, "expires_seconds": 0}, "expires_seconds"),
        ({"size_bytes": 1, "expires_seconds": 3601}, "expires_seconds"),
    ],
)
def test_service_initiate_pdf_upload_rejects_invalid_limits(kwargs, message):
    service, _, _, _, upload_repository = _upload_service()
    session = service.create_session()

    with pytest.raises(ValueError, match=message):
        service.initiate_pdf_upload(
            session.id, expected_sha256="a" * 64, **kwargs
        )

    assert upload_repository.uploads == {}


def test_service_analyze_registered_pdf_blob_skips_workspace_link_after_failed_analysis():
    class FailedHandler(FakeHandler):
        def analyze_paper_input(self, session_id, **kwargs):
            result = super().analyze_paper_input(session_id, **kwargs)
            return result.model_copy(update={"phase": "analysis", "needs_analysis": True})

    handler = FailedHandler()
    blob_store = FakeBlobStore()
    blob_repository = FakeBlobArtifactRepository()
    artifact_repository = FakeArtifactRepository()
    service = PaperIntelService(
        handler=handler, artifact_repository=artifact_repository,
        blob_store=blob_store, blob_artifact_repository=blob_repository,
    )
    session = service.create_session()
    stored = blob_store.put(b"%PDF-1.7\nregistered", kind="pdf")
    artifact = blob_repository.upsert_artifact(stored)
    blob_repository.add_reference(artifact.id, ref_kind="session", ref_id=session.id)

    result = service.analyze_registered_pdf_blob(
        session.id, artifact.id, paper_id="1706.03762"
    )

    assert result.needs_analysis is True
    assert [(ref.ref_kind, ref.ref_id) for ref in blob_repository.references] == [
        ("session", session.id)
    ]
