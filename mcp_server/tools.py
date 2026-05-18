import asyncio
from urllib.parse import urlparse

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
    return (
        "Paper analysis completed.\n\n"
        f"Session ID: {result.session_id}\n"
        f"Phase: {result.phase}\n"
        f"Referenced papers:\n{papers}\n\n"
        f"{result.response_text}\n\n"
        "You can now ask questions with ask_paper."
    )


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
