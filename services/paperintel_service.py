from pathlib import Path
from typing import Protocol

from agents.comparison_analyst import compare_workspaces
from agents.synthesis_agent import synthesize_workspaces
from api.chat_handler import ChatHandler
from models.api import HealthStatus
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.discovery import CandidateStatus, SearchCandidate
from models.jobs import WorkflowJob
from models.session import HandlerResult, Persona, Session, Turn
from models.synthesis import SynthesisAgentResult
from services.selected_candidate_resolver import SelectedCandidateResolver

_FAILED_WORKSPACE_STAGES = {"failed", "paper_failure_finalize"}
MAX_LOCAL_PDF_BYTES = 50 * 1024 * 1024


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

    def latest_comparison(self, session_id: str) -> ComparisonArtifact | None:
        ...

    def save_comparison(self, artifact: ComparisonArtifact) -> ComparisonArtifact:
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
    ) -> None:
        self.handler = handler
        self.health_checker = health_checker
        self.selected_candidate_resolver = selected_candidate_resolver
        self.candidate_repository = candidate_repository
        self.artifact_repository = artifact_repository
        self.workflow_job_repository = workflow_job_repository

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
        resolved_pdf_path = _validate_local_pdf_path(pdf_path)
        content = f"Analyze local PDF {paper_id or resolved_pdf_path}"
        return self.handler.analyze_paper_input(
            session_id,
            input_type="pdf",
            input_value=resolved_pdf_path,
            user_content=content,
            expected_paper_id=paper_id,
            skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
        )

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
