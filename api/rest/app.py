from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.in_memory_session_store import SessionNotFoundError
from api.rest.schemas import (
    AnalyzeRequest,
    AskRequest,
    CompareRequest,
    ComparisonArtifactResponse,
    CreateSessionRequest,
    DiscoverRequest,
    EnqueueAnalyzePaperRequest,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    PaperWorkspaceResponse,
    PaperWorkspacesResponse,
    PaperWorkspaceSummaryResponse,
    SelectPapersRequest,
    SessionResponse,
    SynthesizeRequest,
    TurnsResponse,
    TurnResponse,
    WorkflowJobResponse,
    WorkflowJobsResponse,
)
from services.paperintel_service import (
    ComparisonNotFoundError,
    InvalidPdfInputError,
    InvalidSessionPhaseError,
    InvalidWorkflowJobInputError,
    NoActivePapersError,
    NotEnoughPapersForComparisonError,
    PaperIntelService,
    PaperWorkspaceNotFoundError,
    PaperWorkspaceNotReadyError,
    WorkflowJobNotFoundError,
)
from storage.repositories import InvalidWorkflowJobTransitionError
from services.selected_candidate_resolver import (
    NoSelectedCandidatesError,
    SelectedCandidateNotReadyError,
)

MAX_UPLOAD_PDF_BYTES = 50 * 1024 * 1024


def _pdf_upload_error(status_code: int, error: str, detail: str) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def create_rest_app(*, service: PaperIntelService) -> FastAPI:
    app = FastAPI(title="PaperIntel API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="session_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @app.exception_handler(InvalidSessionPhaseError)
    async def invalid_session_phase_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="invalid_session_phase", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(NoSelectedCandidatesError)
    async def no_selected_candidates_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="no_selected_candidates", detail=str(exc))
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(SelectedCandidateNotReadyError)
    async def selected_candidate_not_ready_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="selected_candidate_not_ready", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(NoActivePapersError)
    async def no_active_papers_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="no_active_papers", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(PaperWorkspaceNotFoundError)
    async def paper_workspace_not_found_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="paper_workspace_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @app.exception_handler(PaperWorkspaceNotReadyError)
    async def paper_workspace_not_ready_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="paper_workspace_not_ready", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(NotEnoughPapersForComparisonError)
    async def not_enough_papers_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="not_enough_papers", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(ComparisonNotFoundError)
    async def comparison_not_found_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="comparison_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @app.exception_handler(WorkflowJobNotFoundError)
    async def workflow_job_not_found_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="workflow_job_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @app.exception_handler(InvalidPdfInputError)
    async def invalid_pdf_input_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="invalid_pdf_input", detail=str(exc))
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(InvalidWorkflowJobInputError)
    async def invalid_workflow_job_input_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="invalid_workflow_job_input", detail=str(exc))
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(InvalidWorkflowJobTransitionError)
    async def invalid_workflow_job_transition_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="invalid_workflow_job_transition", detail=str(exc))
        return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def internal_error_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(
            error="internal_error",
            detail="An internal error occurred while processing the request.",
        )
        return JSONResponse(status_code=500, content=error.model_dump(mode="json"))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        status = service.health()
        response = HealthResponse.from_health_status(status)
        return JSONResponse(
            status_code=200 if status.healthy else 503,
            content=response.model_dump(mode="json"),
        )

    @app.post("/sessions", response_model=SessionResponse)
    async def create_session(payload: CreateSessionRequest):
        session = service.create_session(
            persona=payload.persona,
            original_query=payload.original_query,
        )
        return SessionResponse.from_session(session)

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        session = service.get_session(session_id)
        return SessionResponse.from_session(session)

    @app.get("/sessions/{session_id}/turns", response_model=TurnsResponse)
    async def list_turns(session_id: str, limit: int = 50):
        turns = service.list_turns(session_id, limit=limit)
        return TurnsResponse(
            turns=[TurnResponse.from_turn(turn) for turn in turns],
        )

    @app.get(
        "/sessions/{session_id}/workspaces",
        response_model=PaperWorkspacesResponse,
    )
    async def list_paper_workspaces(session_id: str):
        workspaces = service.list_paper_workspaces(session_id)
        return PaperWorkspacesResponse(
            workspaces=[
                PaperWorkspaceSummaryResponse.from_workspace(workspace)
                for workspace in workspaces
            ],
        )

    @app.get(
        "/sessions/{session_id}/workspaces/{paper_id}",
        response_model=PaperWorkspaceResponse,
    )
    async def get_paper_workspace(session_id: str, paper_id: str):
        workspace = service.get_paper_workspace(session_id, paper_id)
        return PaperWorkspaceResponse.from_workspace(workspace)

    @app.get(
        "/sessions/{session_id}/comparison",
        response_model=ComparisonArtifactResponse,
    )
    async def get_latest_comparison(session_id: str):
        comparison = service.get_latest_comparison(session_id)
        return ComparisonArtifactResponse.from_artifact(comparison)

    @app.post(
        "/sessions/{session_id}/compare",
        response_model=ComparisonArtifactResponse,
    )
    async def compare_papers(
        session_id: str,
        payload: CompareRequest | None = None,
    ):
        """Create a new request-driven comparison artifact.

        Each successful POST runs comparison_analyst over durable paper
        workspaces and persists a new ComparisonArtifact. If paper_ids are not
        provided, the service uses the session active papers.
        """
        artifact = service.compare_papers(
            session_id,
            paper_ids=payload.paper_ids if payload is not None else None,
            prompt=payload.prompt if payload is not None else None,
        )
        return ComparisonArtifactResponse.from_artifact(artifact)


    @app.post("/sessions/{session_id}/analyze-pdf", response_model=MessageResponse)
    async def analyze_pdf_upload(
        session_id: str,
        file: UploadFile = File(...),
        paper_id: str | None = Form(default=None),
        skip_arxiv_metadata_fetch: bool = Form(default=False),
    ):
        content_type = (file.content_type or "").lower()
        if content_type and content_type != "application/pdf":
            return _pdf_upload_error(
                415,
                "unsupported_media_type",
                "PDF upload must use content type application/pdf.",
            )

        data = await file.read(MAX_UPLOAD_PDF_BYTES + 1)
        if len(data) > MAX_UPLOAD_PDF_BYTES:
            return _pdf_upload_error(
                413,
                "pdf_too_large",
                f"PDF upload exceeds {MAX_UPLOAD_PDF_BYTES} bytes.",
            )
        if not data.startswith(b"%PDF-"):
            return _pdf_upload_error(
                415,
                "unsupported_media_type",
                "PDF upload must start with %PDF- magic bytes.",
            )

        temp_path = None
        try:
            suffix = Path(file.filename or "uploaded.pdf").suffix or ".pdf"
            with NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                prefix="paperintel_upload_",
                delete=False,
            ) as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name
            result = service.analyze_pdf(
                session_id,
                temp_path,
                paper_id=paper_id.strip() if paper_id and paper_id.strip() else None,
                skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
            )
            return MessageResponse.from_handler_result(result)
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

    @app.post("/sessions/{session_id}/analyze", response_model=MessageResponse)
    async def analyze_paper(session_id: str, payload: AnalyzeRequest):
        result = service.analyze_paper(session_id, str(payload.paper_url))
        return MessageResponse.from_handler_result(result)

    @app.post("/sessions/{session_id}/ask", response_model=MessageResponse)
    async def ask_question(session_id: str, payload: AskRequest):
        result = service.ask_question(session_id, payload.question)
        return MessageResponse.from_handler_result(result)

    @app.post("/sessions/{session_id}/discover", response_model=MessageResponse)
    async def discover_papers(session_id: str, payload: DiscoverRequest):
        result = service.discover_papers(session_id, payload.topic)
        return MessageResponse.from_handler_result(result)

    @app.post("/sessions/{session_id}/select", response_model=MessageResponse)
    async def select_papers(session_id: str, payload: SelectPapersRequest):
        result = service.select_papers(session_id, payload.selection)
        return MessageResponse.from_handler_result(result)

    @app.post("/sessions/{session_id}/analyze-selected", response_model=MessageResponse)
    async def analyze_selected_papers(session_id: str):
        result = service.analyze_selected_papers(session_id)
        return MessageResponse.from_handler_result(result)

    @app.post(
        "/sessions/{session_id}/jobs/analyze-paper",
        response_model=WorkflowJobResponse,
        status_code=202,
    )
    async def enqueue_analyze_paper_job(
        session_id: str,
        payload: EnqueueAnalyzePaperRequest,
    ):
        job = service.enqueue_analyze_paper(session_id, str(payload.paper_url))
        return WorkflowJobResponse.from_job(job)

    @app.post(
        "/sessions/{session_id}/jobs/analyze-selected",
        response_model=WorkflowJobResponse,
        status_code=202,
    )
    async def enqueue_analyze_selected_job(session_id: str):
        job = service.enqueue_analyze_selected(session_id)
        return WorkflowJobResponse.from_job(job)

    @app.get(
        "/sessions/{session_id}/jobs",
        response_model=WorkflowJobsResponse,
    )
    async def list_workflow_jobs(session_id: str, limit: int = 50):
        jobs = service.list_workflow_jobs(session_id, limit=limit)
        return WorkflowJobsResponse(
            jobs=[WorkflowJobResponse.from_job(job) for job in jobs],
        )

    @app.get("/jobs/{job_id}", response_model=WorkflowJobResponse)
    async def get_workflow_job(job_id: str):
        job = service.get_workflow_job(job_id)
        return WorkflowJobResponse.from_job(job)

    @app.post("/jobs/{job_id}/cancel", response_model=WorkflowJobResponse)
    async def cancel_workflow_job(job_id: str):
        job = service.cancel_workflow_job(job_id)
        return WorkflowJobResponse.from_job(job)

    @app.post("/sessions/{session_id}/synthesize", response_model=MessageResponse)
    async def synthesize_papers(
        session_id: str,
        payload: SynthesizeRequest | None = None,
    ):
        result = service.synthesize_papers(
            session_id,
            prompt=payload.prompt if payload is not None else None,
        )
        return MessageResponse.from_synthesis_result(
            session_id=session_id,
            result=result,
        )

    return app
