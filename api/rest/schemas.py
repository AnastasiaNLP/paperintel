from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from models.api import HealthStatus
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.jobs import WorkflowJob
from models.pdf_uploads import PdfUpload, PdfUploadInitiation
from models.session import HandlerResult, Persona, Session, Turn
from models.synthesis import SynthesisAgentResult


class CreateSessionRequest(BaseModel):
    persona: Persona = "engineer"
    original_query: str | None = None


class AnalyzeRequest(BaseModel):
    paper_url: HttpUrl


class EnqueueAnalyzePaperRequest(BaseModel):
    paper_url: HttpUrl


class InitiatePdfUploadRequest(BaseModel):
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=50 * 1024 * 1024)
    content_type: str = Field(default="application/pdf", min_length=1, max_length=100)
    expires_seconds: int = Field(default=900, ge=60, le=3600)


class EnqueuePdfUploadRequest(BaseModel):
    paper_id: str | None = Field(default=None, max_length=500)
    skip_arxiv_metadata_fetch: bool = False
    pipeline_version: str = Field(default="v1", min_length=1, max_length=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DiscoverRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)


class SelectPapersRequest(BaseModel):
    selection: str = Field(min_length=1, max_length=500)


class SynthesizeRequest(BaseModel):
    prompt: str | None = Field(default=None, max_length=2000)


class CompareRequest(BaseModel):
    paper_ids: list[str] | None = None
    prompt: str | None = Field(default=None, max_length=2000)


class SessionResponse(BaseModel):
    id: str
    persona: Persona
    phase: str
    active_paper_ids: list[str]
    original_query: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_session(cls, session: Session) -> "SessionResponse":
        return cls(
            id=session.id,
            persona=session.persona,
            phase=session.phase,
            active_paper_ids=session.active_paper_ids,
            original_query=session.original_query,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class MessageResponse(BaseModel):
    session_id: str
    response_text: str
    phase: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    comparison_markdown: str | None = None
    needs_analysis: bool = False
    needs_discovery: bool = False
    discovery_topic: str | None = None
    discovery_candidate_count: int | None = None
    selected_candidate_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_handler_result(cls, result: HandlerResult) -> "MessageResponse":
        return cls(
            session_id=result.session_id,
            response_text=result.response_text,
            phase=result.phase,
            intent=result.intent,
            referenced_paper_ids=result.referenced_paper_ids,
            citations=[citation.model_dump(mode="json") for citation in result.citations],
            artifact_refs=result.artifact_refs,
            comparison_markdown=result.comparison_markdown,
            needs_analysis=result.needs_analysis,
            needs_discovery=result.needs_discovery,
            discovery_topic=result.discovery_topic,
            discovery_candidate_count=result.discovery_candidate_count,
            selected_candidate_ids=result.selected_candidate_ids,
        )

    @classmethod
    def from_synthesis_result(
        cls,
        *,
        session_id: str,
        result: SynthesisAgentResult,
    ) -> "MessageResponse":
        referenced_paper_ids = [
            ref.removeprefix("paper_workspace:")
            for ref in result.agent_run.input_refs
            if ref.startswith("paper_workspace:")
        ]
        artifact_refs = [
            ref
            for ref in result.agent_run.input_refs
            if ref.startswith("comparison_artifact:")
        ]
        return cls(
            session_id=session_id,
            response_text=result.response_text,
            phase="qa",
            intent="synthesis",
            referenced_paper_ids=referenced_paper_ids,
            citations=[
                citation.model_dump(mode="json")
                for citation in result.report.citations
            ],
            artifact_refs=artifact_refs,
        )


class TurnResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    intent: str | None = None
    referenced_paper_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def from_turn(cls, turn: Turn) -> "TurnResponse":
        return cls(
            id=turn.id,
            session_id=turn.session_id,
            role=turn.role,
            content=turn.content,
            intent=turn.intent,
            referenced_paper_ids=turn.referenced_paper_ids,
            artifact_refs=turn.artifact_refs,
            created_at=turn.created_at,
        )


class TurnsResponse(BaseModel):
    turns: list[TurnResponse]


class PaperWorkspaceSummaryResponse(BaseModel):
    id: str
    session_id: str
    paper_id: str
    title: str | None = None
    source_url: str
    pipeline_stage: str
    has_finalized_report: bool
    has_method_extraction: bool
    benchmark_count: int
    has_readiness: bool
    has_markdown_report: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_workspace(cls, workspace: PaperWorkspace) -> "PaperWorkspaceSummaryResponse":
        return cls(
            id=workspace.id,
            session_id=workspace.session_id,
            paper_id=workspace.paper_id,
            title=workspace.title,
            source_url=workspace.source_url,
            pipeline_stage=workspace.pipeline_stage,
            has_finalized_report=workspace.finalized_report_json is not None,
            has_method_extraction=workspace.method_extraction_json is not None,
            benchmark_count=len(workspace.benchmarks_json),
            has_readiness=workspace.readiness_json is not None,
            has_markdown_report=workspace.full_markdown_report is not None,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )


class PaperWorkspacesResponse(BaseModel):
    workspaces: list[PaperWorkspaceSummaryResponse]


class PaperWorkspaceResponse(PaperWorkspaceSummaryResponse):
    finalized_report_json: dict | None = None
    method_extraction_json: dict | None = None
    benchmarks_json: list[dict] = Field(default_factory=list)
    readiness_json: dict | None = None
    full_markdown_report: str | None = None

    @classmethod
    def from_workspace(cls, workspace: PaperWorkspace) -> "PaperWorkspaceResponse":
        summary = PaperWorkspaceSummaryResponse.from_workspace(workspace)
        return cls(
            **summary.model_dump(mode="json"),
            finalized_report_json=workspace.finalized_report_json,
            method_extraction_json=workspace.method_extraction_json,
            benchmarks_json=workspace.benchmarks_json,
            readiness_json=workspace.readiness_json,
            full_markdown_report=workspace.full_markdown_report,
        )


class ComparisonArtifactResponse(BaseModel):
    id: str
    session_id: str
    paper_ids: list[str]
    comparison_report_json: dict | None = None
    comparison_markdown: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_artifact(cls, artifact: ComparisonArtifact) -> "ComparisonArtifactResponse":
        return cls(
            id=artifact.id,
            session_id=artifact.session_id,
            paper_ids=artifact.paper_ids,
            comparison_report_json=artifact.comparison_report_json,
            comparison_markdown=artifact.comparison_markdown,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )


class WorkflowJobResponse(BaseModel):
    id: str
    session_id: str
    kind: str
    status: str
    input_json: dict
    result_json: dict | None = None
    error_json: dict | None = None
    attempts: int
    max_attempts: int
    pipeline_version: str
    next_attempt_at: datetime | None = None
    locked_by: str | None = None
    locked_at: datetime | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_job(cls, job: WorkflowJob) -> "WorkflowJobResponse":
        return cls(**job.model_dump(mode="json"))


class WorkflowJobsResponse(BaseModel):
    jobs: list[WorkflowJobResponse]


class PdfUploadResponse(BaseModel):
    id: str
    session_id: str
    blob_id: str | None = None
    object_key: str
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    size_bytes: int | None = None
    content_type: str
    status: str
    expires_at: datetime
    finalized_at: datetime | None = None
    error_json: dict | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_upload(cls, upload: PdfUpload) -> "PdfUploadResponse":
        return cls(**upload.model_dump(mode="json"))


class PdfUploadInitiationResponse(BaseModel):
    upload: PdfUploadResponse
    upload_url: str
    upload_headers: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_initiation(
        cls, initiation: PdfUploadInitiation
    ) -> "PdfUploadInitiationResponse":
        return cls(
            upload=PdfUploadResponse.from_upload(initiation.upload),
            upload_url=initiation.upload_url,
            upload_headers=initiation.upload_headers,
        )


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_health_status(cls, status: HealthStatus) -> "HealthResponse":
        return cls(
            status="healthy" if status.healthy else "degraded",
            checks=status.checks,
        )


class ErrorResponse(BaseModel):
    error: str
    detail: str
