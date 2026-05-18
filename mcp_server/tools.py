import asyncio
from urllib.parse import urlparse

from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.session import HandlerResult, Persona, Session
from services.paperintel_service import PaperIntelService


VALID_PERSONAS: set[str] = {"engineer", "researcher", "techlead"}
MAX_QUESTION_LENGTH = 2000


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
    return format_answer_result(result)


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
