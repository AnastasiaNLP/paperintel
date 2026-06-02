import asyncio
from pathlib import Path
from urllib.parse import urlparse

from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.jobs import WorkflowJob
from models.session import HandlerResult, Persona, Session
from models.synthesis import SynthesisAgentResult
from services.paperintel_service import PaperIntelService


VALID_PERSONAS: set[str] = {"engineer", "researcher", "techlead"}
MAX_QUESTION_LENGTH = 2000
MAX_LOCAL_PDF_BYTES = 50 * 1024 * 1024
MAX_PAPER_ID_LENGTH = 500
MAX_PIPELINE_VERSION_LENGTH = 100


async def create_session_tool(
    service: PaperIntelService,
    *,
    persona: str = "engineer",
) -> str:
    persona = _validate_persona(persona)
    try:
        session = await _run_sync(service.create_session, persona=persona)
    except Exception:
        return _safe_error("create a PaperIntel session")
    return format_session_created(session)


async def analyze_paper_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    paper_url: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    paper_url = _validate_url(paper_url)
    try:
        result = await _run_sync(service.analyze_paper, session_id, paper_url)
    except Exception:
        return _safe_error("analyze the paper")
    return format_analysis_result(result)


async def analyze_pdf_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    pdf_path: str,
    paper_id: str | None = None,
    skip_arxiv_metadata_fetch: bool = False,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    pdf_path = _validate_non_empty("pdf_path", pdf_path)
    if paper_id is not None:
        paper_id = paper_id.strip() or None
    try:
        result = await _run_sync(
            service.analyze_pdf,
            session_id,
            pdf_path,
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=bool(skip_arxiv_metadata_fetch),
        )
    except Exception:
        return _safe_error("analyze the local PDF")
    return format_analysis_result(result)


async def ask_paper_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    question: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    question = _validate_question(question)
    try:
        result = await _run_sync(service.ask_question, session_id, question)
    except Exception:
        return _safe_error("answer the question")
    return format_answer_result(result)


async def discover_papers_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    topic: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    topic = _validate_non_empty("topic", topic)
    try:
        result = await _run_sync(service.discover_papers, session_id, topic)
    except Exception:
        return _safe_error("discover papers")
    return format_discovery_result(result)


async def select_papers_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    selection: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    selection = _validate_non_empty("selection", selection)
    try:
        result = await _run_sync(service.select_papers, session_id, selection)
    except Exception:
        return _safe_error("select papers")
    return format_selection_result(result)


async def analyze_selected_papers_tool(
    service: PaperIntelService,
    *,
    session_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    try:
        result = await _run_sync(service.analyze_selected_papers, session_id)
    except Exception:
        return _safe_error("analyze the selected papers")
    return format_analysis_result(result)


async def enqueue_analyze_paper_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    paper_url: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    paper_url = _validate_url(paper_url)
    try:
        job = await _run_sync(service.enqueue_analyze_paper, session_id, paper_url)
    except Exception:
        return _safe_error("enqueue paper analysis")
    return format_workflow_job(job, heading="Queued paper analysis job")


async def enqueue_analyze_selected_tool(
    service: PaperIntelService,
    *,
    session_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    try:
        job = await _run_sync(service.enqueue_analyze_selected, session_id)
    except Exception:
        return _safe_error("enqueue selected-paper analysis")
    return format_workflow_job(job, heading="Queued selected-paper analysis job")


async def enqueue_analyze_pdf_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    pdf_path: str,
    paper_id: str | None = None,
    skip_arxiv_metadata_fetch: bool = False,
    pipeline_version: str = "v1",
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    paper_id = _validate_optional_text("paper_id", paper_id, MAX_PAPER_ID_LENGTH)
    pipeline_version = _validate_bounded_text(
        "pipeline_version", pipeline_version, MAX_PIPELINE_VERSION_LENGTH
    )
    if not isinstance(skip_arxiv_metadata_fetch, bool):
        raise ValueError("skip_arxiv_metadata_fetch must be a boolean")
    content = await _run_sync(_read_local_pdf, pdf_path)
    try:
        upload = await _run_sync(service.store_pdf_upload, session_id, content)
        job = await _run_sync(
            service.enqueue_analyze_pdf_blob,
            session_id,
            upload.id,
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=skip_arxiv_metadata_fetch,
            pipeline_version=pipeline_version,
        )
    except Exception:
        return _safe_error("enqueue local PDF analysis")
    return format_pdf_workflow_job(
        job, upload_id=str(job.input_json.get("upload_id") or upload.id)
    )


async def get_workflow_job_tool(
    service: PaperIntelService,
    *,
    job_id: str,
) -> str:
    job_id = _validate_non_empty("job_id", job_id)
    try:
        job = await _run_sync(service.get_workflow_job, job_id)
    except Exception:
        return _safe_error("load the workflow job")
    return format_workflow_job(job, heading="Workflow job")


async def list_workflow_jobs_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    limit: int = 50,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    limit = _validate_limit(limit)
    try:
        jobs = await _run_sync(service.list_workflow_jobs, session_id, limit=limit)
    except Exception:
        return _safe_error("list workflow jobs")
    return format_workflow_job_list(jobs)


async def cancel_workflow_job_tool(
    service: PaperIntelService,
    *,
    job_id: str,
) -> str:
    job_id = _validate_non_empty("job_id", job_id)
    try:
        job = await _run_sync(service.cancel_workflow_job, job_id)
    except Exception:
        return _safe_error("cancel the workflow job")
    return format_workflow_job(job, heading="Canceled workflow job")


async def synthesize_papers_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    prompt: str | None = None,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    if prompt is not None:
        prompt = prompt.strip() or None
    if prompt is not None:
        prompt = _validate_question(prompt)
    try:
        result = await _run_sync(service.synthesize_papers, session_id, prompt=prompt)
    except Exception:
        return _safe_error("synthesize the active papers")
    return format_synthesis_result(result)


async def compare_papers_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    paper_ids: list[str] | None = None,
    prompt: str | None = None,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    if prompt is not None:
        prompt = prompt.strip() or None
    if prompt is not None:
        prompt = _validate_question(prompt)
    paper_ids = _validate_optional_paper_ids(paper_ids)
    try:
        artifact = await _run_sync(
            service.compare_papers,
            session_id,
            paper_ids=paper_ids,
            prompt=prompt,
        )
    except Exception:
        return _safe_error("compare the selected papers")
    return format_comparison_artifact(artifact)


async def get_session_tool(
    service: PaperIntelService,
    *,
    session_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    try:
        session = await _run_sync(service.get_session, session_id)
    except Exception:
        return _safe_error("load the session")
    return format_session_state(session)


async def list_paper_workspaces_tool(
    service: PaperIntelService,
    *,
    session_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    try:
        workspaces = await _run_sync(service.list_paper_workspaces, session_id)
    except Exception:
        return _safe_error("load paper workspaces")
    return format_workspace_list(workspaces)


async def get_paper_workspace_tool(
    service: PaperIntelService,
    *,
    session_id: str,
    paper_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    paper_id = _validate_non_empty("paper_id", paper_id)
    try:
        workspace = await _run_sync(service.get_paper_workspace, session_id, paper_id)
    except Exception:
        return _safe_error("load the paper workspace")
    return format_paper_workspace(workspace)


async def get_latest_comparison_tool(
    service: PaperIntelService,
    *,
    session_id: str,
) -> str:
    session_id = _validate_non_empty("session_id", session_id)
    try:
        comparison = await _run_sync(service.get_latest_comparison, session_id)
    except Exception:
        return _safe_error("load the latest comparison")
    return format_comparison_artifact(comparison)


def format_session_created(session: Session) -> str:
    return (
        "Created PaperIntel session.\n\n"
        f"Session ID: {session.id}\n"
        f"Persona: {session.persona}\n\n"
        "Pass this session_id to analyze_paper, ask_paper, or discover_papers."
    )


def format_session_state(session: Session) -> str:
    papers = _format_active_papers(session.active_paper_ids)
    return (
        f"Session: {session.id}\n"
        f"Persona: {session.persona}\n"
        f"Phase: {session.phase}\n"
        f"Active papers:\n{papers}"
    )


def format_analysis_result(result: HandlerResult) -> str:
    papers = _format_active_papers(result.referenced_paper_ids)
    text = (
        "Paper analysis completed.\n\n"
        f"Session ID: {result.session_id}\n"
        f"Phase: {result.phase}\n"
        f"Referenced papers:\n{papers}\n\n"
        f"{result.response_text}\n\n"
        "You can now ask questions with ask_paper."
    )
    if result.comparison_markdown and result.comparison_markdown not in result.response_text:
        text = (
            f"{text}\n\n"
            "Batch comparison report:\n\n"
            f"{result.comparison_markdown}"
        )
    return text


def format_answer_result(result: HandlerResult) -> str:
    text = result.response_text.strip()
    citations = _format_citations(result)
    if citations:
        return f"{text}\n\nSources:\n{citations}"
    return text


def format_synthesis_result(result: SynthesisAgentResult) -> str:
    text = result.response_text.strip()
    if result.report.citations:
        citations = "\n".join(
            f"- {citation.paper_id}: {citation.quote_or_summary}"
            for citation in result.report.citations
        )
        return f"{text}\n\nSources:\n{citations}"
    return text


def format_discovery_result(result: HandlerResult) -> str:
    lines = [result.response_text.strip()]
    if result.discovery_topic:
        lines.append(f"\nTopic: {result.discovery_topic}")
    if result.discovery_candidate_count is not None:
        lines.append(f"Candidates found: {result.discovery_candidate_count}")
    lines.append(f"Session phase: {result.phase}")
    lines.append("\nReply with select_papers using display numbers, for example: 1, 3")
    return "\n".join(line for line in lines if line)


def format_selection_result(result: HandlerResult) -> str:
    text = result.response_text.strip()
    if result.selected_candidate_ids:
        selected = "\n".join(
            f"- {candidate_id}" for candidate_id in result.selected_candidate_ids
        )
        return f"{text}\n\nSelected candidate IDs:\n{selected}"
    return text


def format_workspace_list(workspaces: list[PaperWorkspace]) -> str:
    if not workspaces:
        return "No persisted paper workspaces are available for this session yet."
    lines = ["Persisted paper workspaces:"]
    for workspace in workspaces:
        title = f" - {workspace.title}" if workspace.title else ""
        artifacts = _format_workspace_artifact_flags(workspace)
        lines.append(
            f"- {workspace.paper_id}{title}\n"
            f"  Stage: {workspace.pipeline_stage}\n"
            f"  Artifacts: {artifacts}"
        )
    return "\n".join(lines)


def format_paper_workspace(workspace: PaperWorkspace) -> str:
    lines = [
        f"Paper workspace: {workspace.paper_id}",
        f"Title: {workspace.title or 'unknown'}",
        f"Source: {workspace.source_url}",
        f"Pipeline stage: {workspace.pipeline_stage}",
        f"Artifacts: {_format_workspace_artifact_flags(workspace)}",
    ]
    method = workspace.method_extraction_json or {}
    if method:
        lines.extend(
            [
                "",
                "Method:",
                f"- Name: {method.get('method_name') or 'unknown'}",
                f"- Novelty: {method.get('novelty_claim') or 'not captured'}",
            ]
        )
    readiness = workspace.readiness_json or {}
    if readiness:
        lines.extend(
            [
                "",
                "Production readiness:",
                f"- Maturity: {readiness.get('maturity_level') or 'unknown'}",
                f"- Open code: {readiness.get('has_open_code')}",
            ]
        )
    if workspace.benchmarks_json:
        lines.append("")
        lines.append("Benchmarks:")
        for benchmark in workspace.benchmarks_json[:5]:
            task = benchmark.get("task") or "unknown task"
            metric = benchmark.get("metric") or "metric"
            value = benchmark.get("value")
            lines.append(f"- {task}: {metric}={value}")
    if workspace.full_markdown_report:
        lines.extend(["", "Report:", workspace.full_markdown_report.strip()])
    return "\n".join(lines)


def format_comparison_artifact(artifact: ComparisonArtifact) -> str:
    papers = _format_active_papers(artifact.paper_ids)
    return (
        "Latest persisted comparison\n\n"
        f"Session ID: {artifact.session_id}\n"
        f"Papers:\n{papers}\n\n"
        f"{artifact.comparison_markdown.strip()}"
    )


def format_workflow_job(job: WorkflowJob, *, heading: str = "Workflow job") -> str:
    lines = [
        heading,
        "",
        f"Job ID: {job.id}",
        f"Session ID: {job.session_id}",
        f"Kind: {job.kind}",
        f"Status: {job.status}",
        f"Attempts: {job.attempts}/{job.max_attempts}",
    ]
    if job.locked_by:
        lines.append(f"Locked by: {job.locked_by}")
    if job.result_json is not None:
        lines.extend(["", "Result:", _format_json_summary(job.result_json)])
    if job.error_json is not None:
        lines.extend(["", "Error:", _format_json_summary(job.error_json)])
    return "\n".join(lines)


def format_pdf_workflow_job(job: WorkflowJob, *, upload_id: str) -> str:
    return (
        f"{format_workflow_job(job, heading='Queued local PDF analysis job')}\n"
        f"Upload ID: {upload_id}\n"
        f"Pipeline version: {job.pipeline_version}\n\n"
        f"Poll status with get_workflow_job(job_id='{job.id}')."
    )


def format_workflow_job_list(jobs: list[WorkflowJob]) -> str:
    if not jobs:
        return "No workflow jobs are available for this session yet."
    lines = ["Workflow jobs:"]
    for job in jobs:
        lines.append(
            f"- {job.id}: {job.kind} / {job.status} "
            f"(attempts {job.attempts}/{job.max_attempts})"
        )
    return "\n".join(lines)


def _format_json_summary(payload: dict) -> str:
    if not payload:
        return "{}"
    items = []
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            items.append(f"- {key}: {value}")
        elif isinstance(value, list):
            items.append(f"- {key}: {len(value)} item(s)")
        elif isinstance(value, dict):
            items.append(f"- {key}: object")
        else:
            items.append(f"- {key}: {type(value).__name__}")
    return "\n".join(items)


def _format_workspace_artifact_flags(workspace: PaperWorkspace) -> str:
    flags = []
    if workspace.finalized_report_json is not None or workspace.full_markdown_report:
        flags.append("report")
    if workspace.method_extraction_json is not None:
        flags.append("method")
    if workspace.benchmarks_json:
        flags.append(f"{len(workspace.benchmarks_json)} benchmark(s)")
    if workspace.readiness_json is not None:
        flags.append("readiness")
    return ", ".join(flags) if flags else "none"


def _format_active_papers(paper_ids: list[str]) -> str:
    if not paper_ids:
        return "- none"
    return "\n".join(f"- {paper_id}" for paper_id in paper_ids)


def _format_citations(result: HandlerResult) -> str:
    lines = []
    for citation in result.citations:
        page = _format_page_range(citation.page_start, citation.page_end)
        lines.append(f"- {citation.paper_id}{page}, chunk {citation.chunk_id}")
    return "\n".join(lines)


def _format_page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return f", page {page_start}"
    if page_start is None:
        return f", page {page_end}"
    return f", pages {page_start}-{page_end}"


def _validate_persona(persona: str) -> Persona:
    persona = _validate_non_empty("persona", persona)
    if persona not in VALID_PERSONAS:
        raise ValueError(
            "persona must be one of: engineer, researcher, techlead"
        )
    return persona  # type: ignore[return-value]


def _validate_url(value: str) -> str:
    value = _validate_non_empty("paper_url", value)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("paper_url must be an http or https URL")
    return value


def _validate_question(question: str) -> str:
    question = _validate_non_empty("question", question)
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"question must be at most {MAX_QUESTION_LENGTH} characters")
    return question


def _validate_bounded_text(name: str, value: str, max_length: int) -> str:
    value = _validate_non_empty(name, value)
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    return value


def _validate_optional_text(
    name: str, value: str | None, max_length: int
) -> str | None:
    if value is None:
        return None
    value = value.strip() or None
    if value is not None and len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    return value


def _read_local_pdf(pdf_path: str) -> bytes:
    pdf_path = _validate_non_empty("pdf_path", pdf_path)
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise ValueError("PDF file does not exist")
    if not path.is_file():
        raise ValueError("PDF path is not a file")
    if path.suffix.lower() != ".pdf":
        raise ValueError("PDF path must use a .pdf suffix")
    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_LOCAL_PDF_BYTES + 1)
    except OSError as exc:
        raise ValueError("PDF file could not be read") from exc
    if not content:
        raise ValueError("PDF file must not be empty")
    if len(content) > MAX_LOCAL_PDF_BYTES:
        raise ValueError(f"PDF file must not exceed {MAX_LOCAL_PDF_BYTES} bytes")
    if not content.startswith(b"%PDF-"):
        raise ValueError("PDF file must start with %PDF- magic bytes")
    return content


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    return limit


def _validate_optional_paper_ids(paper_ids: list[str] | None) -> list[str] | None:
    if paper_ids is None:
        return None
    if not isinstance(paper_ids, list):
        raise ValueError("paper_ids must be a list of strings")
    validated = [
        _validate_non_empty("paper_id", paper_id)
        for paper_id in paper_ids
    ]
    deduped = list(dict.fromkeys(validated))
    if len(deduped) < 2:
        raise ValueError("paper_ids must contain at least two distinct paper ids")
    return validated


def _validate_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _safe_error(action: str) -> str:
    return f"PaperIntel could not {action} safely. Please try again."


async def _run_sync(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)
