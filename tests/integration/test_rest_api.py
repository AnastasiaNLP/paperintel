import asyncio
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from api.in_memory_session_store import SessionNotFoundError
from api.rest.app import create_rest_app
from models.agent_runs import AgentRun
from models.api import HealthStatus
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.discovery import SearchCandidate
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
from models.session import HandlerResult, Session, Turn, utc_now
from models.synthesis import (
    SynthesisAgentResult,
    SynthesisCitation,
    SynthesisRecommendation,
    SynthesisReport,
)
from services.paperintel_service import (
    ComparisonNotFoundError,
    InvalidPdfInputError,
    InvalidSessionPhaseError,
    NoActivePapersError,
    NotEnoughPapersForComparisonError,
    PaperWorkspaceNotFoundError,
    PaperWorkspaceNotReadyError,
    WorkflowJobNotFoundError,
)
from services.blob_store import (
    BlobNotFoundError,
    BlobSizeLimitError,
    BlobStoreUnavailableError,
)
from services.selected_candidate_resolver import (
    NoSelectedCandidatesError,
    SelectedCandidateNotReadyError,
)


class FakeService:
    def __init__(self) -> None:
        self.sessions = {
            "session-1": Session(
                id="session-1",
                persona="engineer",
                phase="qa",
                active_paper_ids=["1706.03762"],
            )
        }
        self.turns = [
            Turn(
                id="turn-1",
                session_id="session-1",
                role="user",
                content="What is the contribution?",
                intent="qa_factual",
                referenced_paper_ids=["1706.03762"],
            )
        ]
        self.created_payloads = []
        self.analyze_calls = []
        self.analyze_pdf_calls = []
        self.ask_calls = []
        self.discover_calls = []
        self.select_calls = []
        self.analyze_selected_calls = []
        self.synthesize_calls = []
        self.compare_calls = []
        self.enqueue_analyze_paper_calls = []
        self.enqueue_analyze_selected_calls = []
        self.initiate_pdf_upload_calls = []
        self.finalize_pdf_upload_calls = []
        self.store_pdf_upload_calls = []
        self.enqueue_analyze_pdf_blob_calls = []
        self.workflow_job_list_calls = []
        self.workflow_job_cancel_calls = []
        self.jobs = [
            WorkflowJob(
                id="job-1",
                session_id="session-1",
                kind="analyze_paper",
                status="queued",
                input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
            )
        ]
        self.workspaces = [
            PaperWorkspace(
                session_id="session-1",
                paper_id="1706.03762",
                title="Attention Is All You Need",
                source_url="https://arxiv.org/abs/1706.03762",
                pipeline_stage="completed",
                method_extraction_json={"method_name": "Transformer"},
                benchmarks_json=[{"task": "translation", "metric": "BLEU", "value": 28.4}],
                readiness_json={"maturity_level": "production"},
                full_markdown_report="# Report",
            )
        ]
        self.comparison = ComparisonArtifact(
            session_id="session-1",
            paper_ids=["1706.03762", "2401.00001"],
            comparison_report_json={"winner_basis": "quality"},
            comparison_markdown="# Comparison\n\nA vs B",
        )
        self.health_status = HealthStatus(healthy=True, checks={"basic": "ok"})

    def create_session(self, *, persona="engineer", original_query=None):
        self.created_payloads.append(
            {"persona": persona, "original_query": original_query}
        )
        session = Session(
            id="created-session",
            persona=persona,
            original_query=original_query,
        )
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id):
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"Session not found: {session_id}") from exc

    def list_turns(self, session_id, *, limit=50):
        self.get_session(session_id)
        return self.turns[:limit]

    def analyze_paper(self, session_id, paper_url):
        self.get_session(session_id)
        self.analyze_calls.append((session_id, paper_url))
        return _handler_result(session_id=session_id, response_text="Analyzed paper.")

    def analyze_pdf(
        self,
        session_id,
        pdf_path,
        *,
        paper_id=None,
        skip_arxiv_metadata_fetch=False,
    ):
        self.get_session(session_id)
        self.analyze_pdf_calls.append(
            {
                "session_id": session_id,
                "pdf_path": pdf_path,
                "paper_id": paper_id,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
                "exists_during_call": Path(pdf_path).exists(),
            }
        )
        return _handler_result(session_id=session_id, response_text="Analyzed PDF.")

    def ask_question(self, session_id, question):
        self.get_session(session_id)
        self.ask_calls.append((session_id, question))
        return _handler_result(
            session_id=session_id,
            response_text="The answer.",
            intent="qa_factual",
            referenced_paper_ids=["1706.03762"],
        )

    def discover_papers(self, session_id, topic):
        self.get_session(session_id)
        self.discover_calls.append((session_id, topic))
        return _handler_result(
            session_id=session_id,
            response_text="Here are candidate papers. Reply with numbers.",
            phase="selection",
            intent="discover",
            discovery_topic="agent memory",
            discovery_candidate_count=3,
        )

    def select_papers(self, session_id, selection):
        self.get_session(session_id)
        self.select_calls.append((session_id, selection))
        return _handler_result(
            session_id=session_id,
            response_text="Selected papers 1 and 3.",
            phase="idle",
            intent="select_papers",
            referenced_paper_ids=["2605.1", "2605.3"],
            selected_candidate_ids=["candidate-1", "candidate-3"],
        )

    def analyze_selected_papers(self, session_id):
        self.get_session(session_id)
        self.analyze_selected_calls.append(session_id)
        return _handler_result(
            session_id=session_id,
            response_text="Selected papers analyzed.",
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=["2605.1", "2605.3"],
            comparison_markdown="# Paper Comparison\n\n2605.1 vs 2605.3",
        )

    def enqueue_analyze_paper(self, session_id, paper_url):
        self.get_session(session_id)
        self.enqueue_analyze_paper_calls.append((session_id, paper_url))
        return WorkflowJob(
            id="job-queued-paper",
            session_id=session_id,
            kind="analyze_paper",
            status="queued",
            input_json={"paper_url": paper_url},
        )

    def enqueue_analyze_selected(self, session_id):
        self.get_session(session_id)
        self.enqueue_analyze_selected_calls.append(session_id)
        return WorkflowJob(
            id="job-queued-selected",
            session_id=session_id,
            kind="analyze_selected",
            status="queued",
            input_json={},
        )

    def initiate_pdf_upload(
        self,
        session_id,
        *,
        expected_sha256,
        size_bytes,
        content_type="application/pdf",
        expires_seconds=900,
    ):
        self.get_session(session_id)
        self.initiate_pdf_upload_calls.append(
            {
                "session_id": session_id,
                "expected_sha256": expected_sha256,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "expires_seconds": expires_seconds,
            }
        )
        upload = self._pdf_upload(
            session_id=session_id,
            expected_sha256=expected_sha256,
            size_bytes=size_bytes,
        )
        return PdfUploadInitiation(
            upload=upload,
            upload_url="https://blob.example.test/presigned-upload",
            upload_headers={"Content-Type": content_type},
        )

    def finalize_pdf_upload(self, session_id, upload_id):
        self.get_session(session_id)
        self.finalize_pdf_upload_calls.append((session_id, upload_id))
        return self._pdf_upload(
            session_id=session_id,
            upload_id=upload_id,
            status="finalized",
            blob_id="blob-1",
            expected_sha256="a" * 64,
            actual_sha256="a" * 64,
            size_bytes=128,
        )

    def store_pdf_upload(self, session_id, content):
        self.get_session(session_id)
        self.store_pdf_upload_calls.append((session_id, content))
        return self._pdf_upload(
            session_id=session_id,
            upload_id="upload-stored",
            status="finalized",
            blob_id="blob-stored",
            expected_sha256="a" * 64,
            actual_sha256="a" * 64,
            size_bytes=len(content),
        )

    def enqueue_analyze_pdf_blob(
        self,
        session_id,
        upload_id,
        *,
        paper_id=None,
        skip_arxiv_metadata_fetch=False,
        pipeline_version="v1",
    ):
        self.get_session(session_id)
        self.enqueue_analyze_pdf_blob_calls.append(
            {
                "session_id": session_id,
                "upload_id": upload_id,
                "paper_id": paper_id.strip() if paper_id else None,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
                "pipeline_version": pipeline_version.strip(),
            }
        )
        return WorkflowJob(
            id=f"job-{upload_id}",
            session_id=session_id,
            kind="analyze_pdf_blob",
            status="queued",
            input_json={
                "blob_id": "blob-1",
                "upload_id": upload_id,
                "paper_id": paper_id.strip() if paper_id else None,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
                "pipeline_version": pipeline_version.strip(),
            },
            pipeline_version=pipeline_version.strip(),
        )

    @staticmethod
    def _pdf_upload(
        *,
        session_id,
        upload_id="upload-1",
        status="initiated",
        blob_id=None,
        expected_sha256=None,
        actual_sha256=None,
        size_bytes=None,
    ):
        finalized_at = utc_now() if status in {"finalized", "enqueued"} else None
        return PdfUpload(
            id=upload_id,
            session_id=session_id,
            blob_id=blob_id,
            object_key=f"uploads/{session_id}/{upload_id}.pdf",
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            status=status,
            finalized_at=finalized_at,
            expires_at=utc_now() + timedelta(minutes=15),
        )

    def get_workflow_job(self, job_id):
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise WorkflowJobNotFoundError(job_id)

    def list_workflow_jobs(self, session_id, *, limit=50):
        self.get_session(session_id)
        self.workflow_job_list_calls.append((session_id, limit))
        return self.jobs[:limit]

    def cancel_workflow_job(self, job_id):
        self.workflow_job_cancel_calls.append(job_id)
        return self.jobs[0].model_copy(update={"id": job_id, "status": "canceled"})

    def synthesize_papers(self, session_id, prompt=None):
        self.get_session(session_id)
        self.synthesize_calls.append((session_id, prompt))
        run = AgentRun(
            agent_name="synthesis_agent",
            session_id=session_id,
            input_refs=["paper_workspace:1706.03762"],
        )
        run.complete(output_ref="synthesis_report")
        return SynthesisAgentResult(
            report=SynthesisReport(
                persona="engineer",
                summary="Synthesis answer.",
                key_takeaways=["Transformer is mature."],
                trade_offs=["Quality vs cost."],
                recommended_next_steps=[
                    SynthesisRecommendation(
                        recommendation="Prototype.",
                        reasoning="Readiness is sufficient.",
                    )
                ],
                citations=[
                    SynthesisCitation(
                        paper_id="1706.03762",
                        quote_or_summary="Transformer summary.",
                    )
                ],
            ),
            response_text="Synthesis answer.",
            agent_run=run,
        )

    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        self.get_session(session_id)
        self.compare_calls.append((session_id, paper_ids, prompt))
        return ComparisonArtifact(
            session_id=session_id,
            paper_ids=paper_ids or ["1706.03762", "2401.00001"],
            comparison_report_json={
                "producer": "comparison_analyst",
                "winner_basis": "quality",
            },
            comparison_markdown="# Comparison\n\nA vs B",
        )

    def list_paper_workspaces(self, session_id):
        self.get_session(session_id)
        return [
            workspace
            for workspace in self.workspaces
            if workspace.session_id == session_id
        ]

    def get_paper_workspace(self, session_id, paper_id):
        self.get_session(session_id)
        for workspace in self.workspaces:
            if workspace.session_id == session_id and workspace.paper_id == paper_id:
                return workspace
        raise PaperWorkspaceNotFoundError(session_id=session_id, paper_id=paper_id)

    def get_latest_comparison(self, session_id):
        self.get_session(session_id)
        if self.comparison.session_id == session_id:
            return self.comparison
        raise ComparisonNotFoundError(session_id)

    def health(self):
        return self.health_status


class InvalidPdfService(FakeService):
    def analyze_pdf(self, session_id, pdf_path, **kwargs):
        raise InvalidPdfInputError("bad pdf")


class PdfUploadErrorService(FakeService):
    def __init__(self, error) -> None:
        super().__init__()
        self.error = error

    def finalize_pdf_upload(self, session_id, upload_id):
        raise self.error


class ExplodingService(FakeService):
    def ask_question(self, session_id, question):
        raise RuntimeError("traceback details should not leak")


class WrongPhaseService(FakeService):
    def select_papers(self, session_id, selection):
        raise InvalidSessionPhaseError(expected="selection", actual="idle")


class NoSelectionService(FakeService):
    def analyze_selected_papers(self, session_id):
        raise NoSelectedCandidatesError(session_id)


class CandidateNotReadyService(FakeService):
    def analyze_selected_papers(self, session_id):
        raise SelectedCandidateNotReadyError(
            SearchCandidate(
                id="candidate-1",
                session_id=session_id,
                discovery_turn_id="turn-1",
                display_rank=1,
                status="proposed",
                title="Paper",
                url="https://arxiv.org/abs/2605.1",
            )
        )


class NoActivePapersService(FakeService):
    def synthesize_papers(self, session_id, prompt=None):
        raise NoActivePapersError(session_id)


class NotEnoughPapersService(FakeService):
    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        raise NotEnoughPapersForComparisonError(
            session_id=session_id,
            paper_ids=["1706.03762"],
        )

    def synthesize_papers(self, session_id, prompt=None):
        raise NotEnoughPapersForComparisonError(
            session_id=session_id,
            paper_ids=["1706.03762"],
        )


class NotReadyWorkspaceService(FakeService):
    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        raise PaperWorkspaceNotReadyError(
            session_id=session_id,
            paper_id="1706.03762",
            pipeline_stage="failed",
        )

    def synthesize_papers(self, session_id, prompt=None):
        raise PaperWorkspaceNotReadyError(
            session_id=session_id,
            paper_id="1706.03762",
            pipeline_stage="failed",
        )


class MissingWorkspaceService(FakeService):
    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        raise PaperWorkspaceNotFoundError(
            session_id=session_id,
            paper_id="missing",
        )


class MissingComparisonService(FakeService):
    def get_latest_comparison(self, session_id):
        self.get_session(session_id)
        raise ComparisonNotFoundError(session_id)


def _handler_result(
    *,
    session_id: str = "session-1",
    response_text: str = "OK",
    phase: str = "qa",
    intent: str | None = None,
    referenced_paper_ids: list[str] | None = None,
    discovery_topic: str | None = None,
    discovery_candidate_count: int | None = None,
    selected_candidate_ids: list[str] | None = None,
    comparison_markdown: str | None = None,
) -> HandlerResult:
    return HandlerResult(
        session_id=session_id,
        response_text=response_text,
        phase=phase,
        intent=intent,
        referenced_paper_ids=referenced_paper_ids or [],
        discovery_topic=discovery_topic,
        discovery_candidate_count=discovery_candidate_count,
        selected_candidate_ids=selected_candidate_ids or [],
        comparison_markdown=comparison_markdown,
        user_turn_id="turn-user",
        assistant_turn_id="turn-assistant",
    )


def _request(service, method: str, path: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(
            app=create_rest_app(service=service or FakeService()),
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_create_session_returns_session():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions",
        json={"persona": "researcher", "original_query": "memory agents"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "created-session"
    assert response.json()["persona"] == "researcher"
    assert service.created_payloads == [
        {"persona": "researcher", "original_query": "memory agents"}
    ]


def test_create_session_validates_persona():
    response = _request(None, "POST", "/sessions", json={"persona": "manager"})

    assert response.status_code == 422


def test_get_session_returns_session():
    response = _request(None, "GET", "/sessions/session-1")

    assert response.status_code == 200
    assert response.json()["active_paper_ids"] == ["1706.03762"]


def test_get_session_returns_404_for_missing_session():
    response = _request(None, "GET", "/sessions/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found"


def test_list_turns_returns_turns():
    response = _request(None, "GET", "/sessions/session-1/turns?limit=1")

    assert response.status_code == 200
    assert response.json()["turns"][0]["content"] == "What is the contribution?"
    assert response.json()["turns"][0]["referenced_paper_ids"] == ["1706.03762"]


def test_list_paper_workspaces_returns_summaries():
    response = _request(None, "GET", "/sessions/session-1/workspaces")

    assert response.status_code == 200
    workspace = response.json()["workspaces"][0]
    assert workspace["paper_id"] == "1706.03762"
    assert workspace["title"] == "Attention Is All You Need"
    assert workspace["has_method_extraction"] is True
    assert workspace["benchmark_count"] == 1
    assert "full_markdown_report" not in workspace


def test_get_paper_workspace_returns_full_artifact_snapshot():
    response = _request(None, "GET", "/sessions/session-1/workspaces/1706.03762")

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_id"] == "1706.03762"
    assert payload["method_extraction_json"]["method_name"] == "Transformer"
    assert payload["benchmarks_json"][0]["metric"] == "BLEU"
    assert payload["full_markdown_report"] == "# Report"


def test_get_paper_workspace_returns_404_when_missing():
    response = _request(None, "GET", "/sessions/session-1/workspaces/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "paper_workspace_not_found"


def test_get_latest_comparison_returns_artifact():
    response = _request(None, "GET", "/sessions/session-1/comparison")

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_ids"] == ["1706.03762", "2401.00001"]
    assert payload["comparison_markdown"] == "# Comparison\n\nA vs B"


def test_get_latest_comparison_returns_404_when_missing():
    response = _request(MissingComparisonService(), "GET", "/sessions/session-1/comparison")

    assert response.status_code == 404
    assert response.json()["error"] == "comparison_not_found"


def test_compare_creates_new_comparison_with_prompt_and_paper_ids():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/compare",
        json={
            "paper_ids": ["2401.00001", "1706.03762"],
            "prompt": "Prefer deployability.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_ids"] == ["2401.00001", "1706.03762"]
    assert payload["comparison_report_json"]["producer"] == "comparison_analyst"
    assert payload["comparison_markdown"] == "# Comparison\n\nA vs B"
    assert service.compare_calls == [
        ("session-1", ["2401.00001", "1706.03762"], "Prefer deployability.")
    ]


def test_compare_accepts_empty_body_and_uses_active_papers():
    service = FakeService()

    response = _request(service, "POST", "/sessions/session-1/compare")

    assert response.status_code == 200
    assert response.json()["paper_ids"] == ["1706.03762", "2401.00001"]
    assert service.compare_calls == [("session-1", None, None)]


def test_compare_maps_shared_workspace_errors():
    cases = [
        (NotEnoughPapersService(), "not_enough_papers", 409),
        (MissingWorkspaceService(), "paper_workspace_not_found", 404),
        (NotReadyWorkspaceService(), "paper_workspace_not_ready", 409),
    ]

    for service, expected_error, expected_status in cases:
        response = _request(service, "POST", "/sessions/session-1/compare")

        assert response.status_code == expected_status
        assert response.json()["error"] == expected_error


def test_enqueue_analyze_paper_job_endpoint_returns_202():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-paper",
        json={"paper_url": "https://arxiv.org/abs/1706.03762"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["id"] == "job-queued-paper"
    assert payload["kind"] == "analyze_paper"
    assert payload["status"] == "queued"
    assert payload["input_json"] == {"paper_url": "https://arxiv.org/abs/1706.03762"}
    assert service.enqueue_analyze_paper_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]


def test_enqueue_analyze_selected_job_endpoint_returns_202():
    service = FakeService()

    response = _request(service, "POST", "/sessions/session-1/jobs/analyze-selected")

    assert response.status_code == 202
    payload = response.json()
    assert payload["id"] == "job-queued-selected"
    assert payload["kind"] == "analyze_selected"
    assert payload["status"] == "queued"
    assert payload["input_json"] == {}
    assert service.enqueue_analyze_selected_calls == ["session-1"]


def test_workflow_job_status_endpoints_get_list_and_cancel():
    service = FakeService()

    listed = _request(service, "GET", "/sessions/session-1/jobs?limit=10")
    loaded = _request(service, "GET", "/jobs/job-1")
    canceled = _request(service, "POST", "/jobs/job-1/cancel")

    assert listed.status_code == 200
    assert listed.json()["jobs"][0]["id"] == "job-1"
    assert listed.json()["jobs"][0]["kind"] == "analyze_paper"
    assert loaded.status_code == 200
    assert loaded.json()["status"] == "queued"
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    assert service.workflow_job_list_calls == [("session-1", 10)]
    assert service.workflow_job_cancel_calls == ["job-1"]


def test_workflow_job_get_missing_maps_404():
    response = _request(FakeService(), "GET", "/jobs/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "workflow_job_not_found"


def test_initiate_pdf_upload_returns_presigned_contract():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads",
        json={
            "expected_sha256": "a" * 64,
            "size_bytes": 128,
            "content_type": "application/pdf",
            "expires_seconds": 600,
        },
    )

    assert response.status_code == 201
    assert response.json()["upload"]["id"] == "upload-1"
    assert response.json()["upload_url"] == "https://blob.example.test/presigned-upload"
    assert response.json()["upload_headers"] == {"Content-Type": "application/pdf"}
    assert service.initiate_pdf_upload_calls == [
        {
            "session_id": "session-1",
            "expected_sha256": "a" * 64,
            "size_bytes": 128,
            "content_type": "application/pdf",
            "expires_seconds": 600,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_sha256": "not-a-digest", "size_bytes": 128},
        {"expected_sha256": "a" * 64, "size_bytes": 50 * 1024 * 1024 + 1},
        {"expected_sha256": "a" * 64, "size_bytes": 128, "expires_seconds": 59},
        {"expected_sha256": "a" * 64, "size_bytes": 128, "expires_seconds": 3601},
        {"expected_sha256": "a" * 64, "size_bytes": 128, "content_type": "x" * 101},
    ],
)
def test_initiate_pdf_upload_validates_direct_upload_contract(payload):
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads",
        json=payload,
    )

    assert response.status_code == 422
    assert service.initiate_pdf_upload_calls == []


def test_finalize_pdf_upload_returns_durable_upload():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/finalize",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "finalized"
    assert response.json()["blob_id"] == "blob-1"
    assert service.finalize_pdf_upload_calls == [("session-1", "upload-1")]


def test_enqueue_pdf_upload_job_returns_202_and_analysis_options():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/jobs/analyze",
        json={
            "paper_id": "  local-paper  ",
            "skip_arxiv_metadata_fetch": True,
            "pipeline_version": "  pipeline-v2  ",
        },
    )

    assert response.status_code == 202
    assert response.json()["id"] == "job-upload-1"
    assert response.json()["kind"] == "analyze_pdf_blob"
    assert response.json()["pipeline_version"] == "pipeline-v2"
    assert response.json()["next_attempt_at"] is None
    assert response.json()["cancel_requested_at"] is None
    assert service.enqueue_analyze_pdf_blob_calls == [
        {
            "session_id": "session-1",
            "upload_id": "upload-1",
            "paper_id": "local-paper",
            "skip_arxiv_metadata_fetch": True,
            "pipeline_version": "pipeline-v2",
        }
    ]


def test_enqueue_pdf_upload_job_duplicate_returns_same_job():
    service = FakeService()

    first = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/jobs/analyze",
    )
    second = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/jobs/analyze",
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] == second.json()["id"] == "job-upload-1"


def test_enqueue_pdf_upload_job_rejects_oversized_analysis_metadata():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/jobs/analyze",
        json={"paper_id": "x" * 501, "pipeline_version": "y" * 101},
    )

    assert response.status_code == 422
    assert service.enqueue_analyze_pdf_blob_calls == []


def test_enqueue_pdf_multipart_stores_upload_and_returns_job_without_sync_analysis():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
        data={
            "paper_id": "  local-paper  ",
            "skip_arxiv_metadata_fetch": "true",
            "pipeline_version": "  pipeline-v2  ",
        },
    )

    assert response.status_code == 202
    assert response.json()["id"] == "job-upload-stored"
    assert service.store_pdf_upload_calls == [("session-1", b"%PDF-1.7\nbody")]
    assert service.enqueue_analyze_pdf_blob_calls[0] == {
        "session_id": "session-1",
        "upload_id": "upload-stored",
        "paper_id": "local-paper",
        "skip_arxiv_metadata_fetch": True,
        "pipeline_version": "pipeline-v2",
    }
    assert service.analyze_pdf_calls == []


def test_enqueue_pdf_multipart_rejects_non_pdf_content_type():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-pdf",
        files={"file": ("paper.txt", b"%PDF-1.7\nbody", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"
    assert service.store_pdf_upload_calls == []


def test_enqueue_pdf_multipart_rejects_bad_magic_bytes():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-pdf",
        files={"file": ("paper.pdf", b"not pdf", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"
    assert service.store_pdf_upload_calls == []


def test_enqueue_pdf_multipart_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr("api.rest.app.MAX_UPLOAD_PDF_BYTES", 4)
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "pdf_too_large"
    assert service.store_pdf_upload_calls == []


def test_enqueue_pdf_multipart_rejects_oversized_analysis_metadata():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/jobs/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
        data={"paper_id": "x" * 501, "pipeline_version": "y" * 101},
    )

    assert response.status_code == 422
    assert service.store_pdf_upload_calls == []


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (PdfUploadNotFoundError("upload-1"), 404, "pdf_upload_not_found"),
        (PdfUploadExpiredError("upload-1"), 410, "pdf_upload_expired"),
        (
            PdfUploadStateError(
                upload_id="upload-1", status="failed", target_status="finalized"
            ),
            409,
            "pdf_upload_invalid_state",
        ),
        (PdfUploadChecksumMismatchError("bad checksum"), 422, "pdf_upload_checksum_mismatch"),
        (PdfUploadSizeMismatchError("bad size"), 422, "pdf_upload_size_mismatch"),
        (PdfUploadInvalidContentError("bad pdf"), 415, "unsupported_media_type"),
        (BlobStoreUnavailableError("blob down"), 503, "blob_store_unavailable"),
        (BlobNotFoundError("staging object missing"), 404, "blob_not_found"),
        (BlobSizeLimitError("staging object too large"), 422, "pdf_upload_size_mismatch"),
    ],
)
def test_pdf_upload_domain_errors_map_to_stable_http_contract(
    error, status_code, error_code
):
    response = _request(
        PdfUploadErrorService(error),
        "POST",
        "/sessions/session-1/pdf-uploads/upload-1/finalize",
    )

    assert response.status_code == status_code
    assert response.json()["error"] == error_code


def test_health_returns_service_health():
    response = _request(None, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "checks": {"basic": "ok"}}


def test_health_returns_503_when_unhealthy():
    service = FakeService()
    service.health_status = HealthStatus(
        healthy=False,
        checks={"postgres": "error:RuntimeError"},
    )

    response = _request(service, "GET", "/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_analyze_requires_valid_url():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/analyze",
        json={"paper_url": "arxiv 1706.03762"},
    )

    assert response.status_code == 422


def test_analyze_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/analyze",
        json={"paper_url": "https://arxiv.org/abs/1706.03762"},
    )

    assert response.status_code == 200
    assert response.json()["response_text"] == "Analyzed paper."
    assert service.analyze_calls[0][0] == "session-1"
    assert service.analyze_calls[0][1].startswith("https://arxiv.org/abs/1706.03762")


def test_analyze_pdf_upload_returns_message_and_cleans_temp_file():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
        data={"paper_id": "2501.12948", "skip_arxiv_metadata_fetch": "true"},
    )

    assert response.status_code == 200
    assert response.json()["response_text"] == "Analyzed PDF."
    assert len(service.analyze_pdf_calls) == 1
    call = service.analyze_pdf_calls[0]
    assert call["session_id"] == "session-1"
    assert call["paper_id"] == "2501.12948"
    assert call["skip_arxiv_metadata_fetch"] is True
    assert call["exists_during_call"] is True
    assert not Path(call["pdf_path"]).exists()


def test_analyze_pdf_upload_rejects_non_pdf_content_type():
    response = _request(
        FakeService(),
        "POST",
        "/sessions/session-1/analyze-pdf",
        files={"file": ("paper.txt", b"%PDF-1.7\nbody", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"


def test_analyze_pdf_upload_rejects_bad_magic_bytes():
    response = _request(
        FakeService(),
        "POST",
        "/sessions/session-1/analyze-pdf",
        files={"file": ("paper.pdf", b"not pdf", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"


def test_analyze_pdf_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr("api.rest.app.MAX_UPLOAD_PDF_BYTES", 4)

    response = _request(
        FakeService(),
        "POST",
        "/sessions/session-1/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "pdf_too_large"


def test_analyze_pdf_upload_maps_service_pdf_validation_error():
    response = _request(
        InvalidPdfService(),
        "POST",
        "/sessions/session-1/analyze-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.7\nbody", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_pdf_input"


def test_ask_requires_non_empty_question():
    response = _request(None, "POST", "/sessions/session-1/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/ask",
        json={"question": "What is the contribution?"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "qa_factual"
    assert response.json()["referenced_paper_ids"] == ["1706.03762"]
    assert service.ask_calls == [("session-1", "What is the contribution?")]


def test_discover_requires_non_empty_topic():
    response = _request(None, "POST", "/sessions/session-1/discover", json={"topic": ""})

    assert response.status_code == 422


def test_discover_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/discover",
        json={"topic": "Find papers about agent memory"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "discover"
    assert payload["phase"] == "selection"
    assert payload["discovery_topic"] == "agent memory"
    assert payload["discovery_candidate_count"] == 3
    assert service.discover_calls == [
        ("session-1", "Find papers about agent memory")
    ]


def test_discover_accepts_bare_topic():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/discover",
        json={"topic": "agent memory"},
    )

    assert response.status_code == 200
    assert service.discover_calls == [("session-1", "agent memory")]


def test_select_requires_non_empty_selection():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/select",
        json={"selection": ""},
    )

    assert response.status_code == 422


def test_select_calls_service():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/select",
        json={"selection": "use 1 and 3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "select_papers"
    assert payload["phase"] == "idle"
    assert payload["referenced_paper_ids"] == ["2605.1", "2605.3"]
    assert payload["selected_candidate_ids"] == ["candidate-1", "candidate-3"]
    assert service.select_calls == [("session-1", "use 1 and 3")]


def test_select_returns_409_when_session_not_in_selection_phase():
    response = _request(
        WrongPhaseService(),
        "POST",
        "/sessions/session-1/select",
        json={"selection": "use 1"},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "invalid_session_phase"


def test_analyze_selected_calls_service_without_body():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "analyze_paper"
    assert payload["phase"] == "qa"
    assert payload["referenced_paper_ids"] == ["2605.1", "2605.3"]
    assert payload["comparison_markdown"] == "# Paper Comparison\n\n2605.1 vs 2605.3"
    assert service.analyze_selected_calls == ["session-1"]


def test_analyze_selected_returns_400_when_no_candidates_selected():
    response = _request(
        NoSelectionService(),
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "no_selected_candidates"


def test_analyze_selected_returns_409_when_candidate_not_ready():
    response = _request(
        CandidateNotReadyService(),
        "POST",
        "/sessions/session-1/analyze-selected",
    )

    assert response.status_code == 409
    assert response.json()["error"] == "selected_candidate_not_ready"


def test_synthesize_calls_service_with_optional_prompt():
    service = FakeService()

    response = _request(
        service,
        "POST",
        "/sessions/session-1/synthesize",
        json={"prompt": "Compare implementation trade-offs."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "synthesis"
    assert payload["response_text"] == "Synthesis answer."
    assert payload["referenced_paper_ids"] == ["1706.03762"]
    assert payload["citations"][0]["paper_id"] == "1706.03762"
    assert service.synthesize_calls == [
        ("session-1", "Compare implementation trade-offs.")
    ]


def test_synthesize_accepts_empty_body():
    service = FakeService()

    response = _request(service, "POST", "/sessions/session-1/synthesize")

    assert response.status_code == 200
    assert service.synthesize_calls == [("session-1", None)]


def test_synthesize_returns_409_when_no_active_papers():
    response = _request(
        NoActivePapersService(),
        "POST",
        "/sessions/session-1/synthesize",
    )

    assert response.status_code == 409
    assert response.json()["error"] == "no_active_papers"


def test_synthesize_maps_shared_workspace_errors_to_409():
    for service, expected_error in [
        (NotEnoughPapersService(), "not_enough_papers"),
        (NotReadyWorkspaceService(), "paper_workspace_not_ready"),
    ]:
        response = _request(service, "POST", "/sessions/session-1/synthesize")

        assert response.status_code == 409
        assert response.json()["error"] == expected_error


def test_synthesize_rejects_prompt_over_max_length():
    response = _request(
        FakeService(),
        "POST",
        "/sessions/session-1/synthesize",
        json={"prompt": "x" * 2001},
    )

    assert response.status_code == 422


def test_message_response_shape_excludes_internal_fields():
    response = _request(
        None,
        "POST",
        "/sessions/session-1/ask",
        json={"question": "What is the contribution?"},
    )

    payload = response.json()
    assert "agent_runs" not in payload
    assert "errors" not in payload
    assert "user_turn_id" not in payload
    assert "assistant_turn_id" not in payload


def test_internal_error_returns_safe_500_without_traceback_leak():
    response = _request(
        ExplodingService(),
        "POST",
        "/sessions/session-1/ask",
        json={"question": "Boom?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "detail": "An internal error occurred while processing the request.",
    }
    assert "traceback" not in response.text.lower()
    assert "should not leak" not in response.text
