import re
from typing import Any, Protocol

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from api.session_store import SessionStore
from models.errors import ErrorCodes, StructuredError, make_error
from models.discovery import SelectionAdvice
from models.qa import AnswerDraft
from models.session import GraphInvocationResult, HandlerResult, Persona, Session
from services.retrieval_layer import RetrievalLayer
from services.searcher import Searcher
from services.selection_parser import SelectionHandler


class ConversationRunner(Protocol):
    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        ...


class AnalysisRunner(Protocol):
    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        ...


class DiscoveryRunner(Protocol):
    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        ...


_ARXIV_URL_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/[^\s]+")
_GENERIC_PDF_RE = re.compile(r"https?://[^\s]+\.pdf(?:\?[^\s]*)?")
_DISCOVERY_RE = re.compile(
    r"\b(find|search|discover|recommend)\b.*\b(papers?|literature|research)\b"
    r"|\b(recent|new|latest)\b.*\b(papers?|literature|research)\b",
    re.IGNORECASE,
)


class ChatHandler:
    def __init__(
        self,
        *,
        store: SessionStore,
        conversation_runner: ConversationRunner,
        analysis_runner: AnalysisRunner | None = None,
        discovery_runner: DiscoveryRunner | None = None,
        agent_run_persistence: AgentRunPersistence | None = None,
        retrieval_layer: RetrievalLayer | None = None,
        searcher: Searcher | None = None,
        selection_handler: SelectionHandler | None = None,
    ) -> None:
        self.store = store
        self.conversation_runner = conversation_runner
        self.analysis_runner = analysis_runner
        self.discovery_runner = discovery_runner
        self.agent_run_persistence = (
            agent_run_persistence or NoopAgentRunPersistence()
        )
        self.retrieval_layer = retrieval_layer
        self.searcher = searcher
        self.selection_handler = selection_handler

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        return self.store.create_session(
            persona=persona,
            original_query=original_query,
        )

    def handle_message(self, session_id: str, message: str) -> HandlerResult:
        session = self.store.require_session(session_id)
        user_turn = self.store.append_turn(
            session.id,
            role="user",
            content=message,
        )

        try:
            graph_result = self._route_message(
                session,
                message,
                user_turn_id=user_turn.id,
            )
        except Exception as exc:
            error = make_error(
                ErrorCodes.FATAL_ERROR,
                f"Chat routing failed: {exc}",
                node="chat_handler",
                severity="error",
                recoverable=True,
                session_id=session.id,
                exception_type=type(exc).__name__,
            )
            self.store.update_phase(session.id, "failed")
            assistant_turn = self.store.append_turn(
                session.id,
                role="assistant",
                content="I could not complete the request safely. Please try again.",
                error=error,
            )
            return HandlerResult(
                session_id=session.id,
                response_text=assistant_turn.content,
                phase="failed",
                errors=[error],
                user_turn_id=user_turn.id,
                assistant_turn_id=assistant_turn.id,
                error=error,
            )

        if graph_result.next_phase is not None:
            session = self.store.update_phase(session.id, graph_result.next_phase)
        else:
            session = self.store.require_session(session.id)

        assistant_turn = self.store.append_turn(
            session.id,
            role="assistant",
            content=graph_result.response_text,
            intent=graph_result.intent,
            referenced_paper_ids=graph_result.referenced_paper_ids,
            artifact_refs=graph_result.artifact_refs,
        )

        return HandlerResult(
            session_id=session.id,
            response_text=graph_result.response_text,
            phase=session.phase,
            intent=graph_result.intent,
            referenced_paper_ids=graph_result.referenced_paper_ids,
            citations=graph_result.citations,
            artifact_refs=graph_result.artifact_refs,
            needs_analysis=graph_result.needs_analysis,
            needs_discovery=graph_result.needs_discovery,
            discovery_topic=graph_result.discovery_topic,
            discovery_candidate_count=graph_result.discovery_candidate_count,
            selected_candidate_ids=graph_result.selected_candidate_ids,
            agent_runs=graph_result.agent_runs,
            errors=graph_result.errors,
            user_turn_id=user_turn.id,
            assistant_turn_id=assistant_turn.id,
            error=graph_result.errors[0] if graph_result.errors else None,
        )

    def _route_message(
        self,
        session: Session,
        message: str,
        *,
        user_turn_id: str,
    ) -> GraphInvocationResult:
        if session.phase == "selection":
            return self._handle_selection(session, message)

        url = _extract_paper_url(message)
        if url is not None:
            return self._invoke_analysis(session, url)
        if _looks_like_discovery_request(message):
            return self._invoke_discovery(
                session,
                message,
                discovery_turn_id=user_turn_id,
            )

        result = self._invoke_conversation(session, message)
        if result.needs_discovery:
            return self._invoke_discovery(
                session,
                message,
                discovery_topic=result.discovery_topic,
                discovery_turn_id=user_turn_id,
            )
        return result

    def _handle_selection(self, session: Session, message: str) -> GraphInvocationResult:
        if self.selection_handler is None:
            return GraphInvocationResult(
                response_text=(
                    "Paper selection is not configured yet. Please ask for a new "
                    "paper search or send a paper URL directly."
                ),
                intent="select_papers",
                next_phase=session.phase,
                raw={"selection_handler_missing": True},
            )

        result = self.selection_handler.handle(session_id=session.id, message=message)
        if result.selection is None:
            return GraphInvocationResult(
                response_text=result.response_text,
                intent="select_papers",
                next_phase="selection",
                raw={
                    "selection_errors": result.errors,
                    "selected_candidate_ids": [],
                },
            )

        return GraphInvocationResult(
            response_text=result.response_text,
            intent="select_papers",
            referenced_paper_ids=[
                candidate.arxiv_id
                for candidate in result.candidates
                if candidate.arxiv_id is not None
            ],
            selected_candidate_ids=list(result.selection.selected_candidate_ids),
            next_phase="idle",
            raw={
                "selection": result.selection.model_dump(mode="json"),
                "selected_candidate_ids": result.selection.selected_candidate_ids,
                "selected_display_ranks": result.selection.display_ranks,
            },
        )

    def _invoke_conversation(
        self,
        session: Session,
        message: str,
    ) -> GraphInvocationResult:
        raw = self.conversation_runner.invoke(
            {
                "session_id": session.id,
                "user_message": message,
                "persona": session.persona,
                "referenced_paper_ids": list(session.active_paper_ids),
                "agent_runs": [],
                "errors": [],
            },
            config=self._graph_config(session),
        )
        return _normalize_conversation_result(raw)

    def _invoke_analysis(self, session: Session, url: str) -> GraphInvocationResult:
        if self.analysis_runner is None:
            return GraphInvocationResult(
                response_text="Please send a paper URL after analysis is configured.",
                intent="analyze_paper",
                needs_analysis=True,
                next_phase=session.phase,
                raw={"needs_analysis": True, "analysis_runner_missing": True},
            )

        raw = self.analysis_runner.invoke(
            _initial_analysis_state(url),
            config=self._graph_config(session),
        )
        return _normalize_analysis_result(raw)

    def _invoke_discovery(
        self,
        session: Session,
        message: str,
        *,
        discovery_topic: str | None = None,
        discovery_turn_id: str,
    ) -> GraphInvocationResult:
        if self.discovery_runner is None:
            return GraphInvocationResult(
                response_text=(
                    "Discovery is not configured yet. Send a paper URL directly, "
                    "or try again after discovery is enabled."
                ),
                intent="discover",
                needs_discovery=True,
                discovery_topic=discovery_topic or message,
                next_phase=session.phase,
                raw={"needs_discovery": True, "discovery_runner_missing": True},
            )

        raw = self.discovery_runner.invoke(
            {
                "session_id": session.id,
                "user_message": message,
                "persona": session.persona,
                "discovery_topic": discovery_topic or message,
                "discovery_turn_id": discovery_turn_id,
                "agent_runs": [],
                "errors": [],
            },
            config=self._graph_config(session),
        )
        return _normalize_discovery_result(raw)

    def _graph_config(self, session: Session) -> dict[str, Any]:
        configurable: dict[str, Any] = {
            "session_id": session.id,
            "session_store": self.store,
            "agent_run_persistence": self.agent_run_persistence,
        }
        if self.retrieval_layer is not None:
            configurable["retrieval_layer"] = self.retrieval_layer
        if self.searcher is not None:
            configurable["searcher"] = self.searcher
        return {"configurable": configurable}


def _normalize_conversation_result(raw: dict[str, Any]) -> GraphInvocationResult:
    answer_draft = raw.get("answer_draft")
    if isinstance(answer_draft, AnswerDraft):
        response_text = answer_draft.answer_text
        citations = list(answer_draft.citations)
    elif raw.get("needs_analysis"):
        response_text = str(
            raw.get("clarification_question")
            or "Please send the paper URL directly so I can analyze it."
        )
        citations = []
    elif raw.get("needs_discovery"):
        response_text = str(
            raw.get("clarification_question")
            or "I can search for papers on that topic once discovery is configured."
        )
        citations = []
    elif raw.get("clarification_question"):
        response_text = str(raw["clarification_question"])
        citations = []
    elif raw.get("errors"):
        response_text = "I could not complete that request safely. Please try again."
        citations = []
    else:
        response_text = str(
            raw.get("response_text")
            or "I am not sure how to respond. Could you rephrase?"
        )
        citations = []

    return GraphInvocationResult(
        response_text=response_text,
        intent=raw.get("intent"),
        referenced_paper_ids=list(raw.get("referenced_paper_ids") or []),
        citations=citations,
        artifact_refs=list(raw.get("artifact_refs") or []),
        needs_analysis=bool(raw.get("needs_analysis", False)),
        needs_discovery=bool(raw.get("needs_discovery", False)),
        discovery_topic=raw.get("discovery_topic"),
        discovery_candidate_count=None,
        selected_candidate_ids=[],
        agent_runs=list(raw.get("agent_runs") or []),
        errors=_structured_errors(raw.get("errors") or []),
        next_phase=raw.get("next_phase"),
        raw=raw,
    )


def _normalize_analysis_result(raw: dict[str, Any]) -> GraphInvocationResult:
    response_text = str(
        raw.get("full_markdown_report")
        or raw.get("comparison_markdown")
        or raw.get("response_text")
        or "Paper analysis completed."
    )
    referenced_paper_ids = _analysis_referenced_paper_ids(raw)
    return GraphInvocationResult(
        response_text=response_text,
        intent="analyze_paper",
        referenced_paper_ids=referenced_paper_ids,
        agent_runs=list(raw.get("agent_runs") or []),
        errors=_structured_errors(raw.get("errors") or []),
        next_phase=raw.get("next_phase") or "qa",
        raw=raw,
    )


def _analysis_referenced_paper_ids(raw: dict[str, Any]) -> list[str]:
    metadata = raw.get("metadata")
    arxiv_id = getattr(metadata, "arxiv_id", None)
    if arxiv_id:
        return [str(arxiv_id)]

    papers = raw.get("papers") or []
    ids = []
    for paper in papers:
        paper_id = getattr(paper, "paper_id", None) or getattr(paper, "arxiv_id", None)
        if paper_id:
            ids.append(str(paper_id))
    return ids


def _normalize_discovery_result(raw: dict[str, Any]) -> GraphInvocationResult:
    advice = raw.get("selection_advice")
    if isinstance(advice, SelectionAdvice):
        response_text = advice.response_text
    else:
        response_text = str(
            raw.get("response_text")
            or "I could not prepare a paper shortlist. Please try a more specific topic."
        )

    return GraphInvocationResult(
        response_text=response_text,
        intent="discover",
        needs_discovery=False,
        discovery_topic=raw.get("discovery_topic"),
        discovery_candidate_count=_discovery_candidate_count(raw),
        selected_candidate_ids=[],
        agent_runs=list(raw.get("agent_runs") or []),
        errors=_structured_errors(raw.get("errors") or []),
        next_phase=raw.get("next_phase") or "selection",
        raw=raw,
    )


def _discovery_candidate_count(raw: dict[str, Any]) -> int | None:
    advice = raw.get("selection_advice")
    if isinstance(advice, SelectionAdvice):
        return advice.candidate_count
    candidates = raw.get("search_candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return None


def _structured_errors(errors: list[Any]) -> list[StructuredError]:
    return [error for error in errors if isinstance(error, StructuredError)]


def _extract_paper_url(message: str) -> str | None:
    match = _ARXIV_URL_RE.search(message)
    if match:
        return match.group(0).rstrip(".,)")

    match = _GENERIC_PDF_RE.search(message)
    if match:
        return match.group(0).rstrip(".,)")

    return None


def _looks_like_discovery_request(message: str) -> bool:
    return _DISCOVERY_RE.search(message) is not None


def _initial_analysis_state(url: str) -> dict[str, Any]:
    return {
        "input_type": "url",
        "input_value": url,
        "batch_urls": None,
        "papers": [],
        "metadata": None,
        "raw_text": None,
        "pdf_path": None,
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
        "ingestion_provenance": None,
        "comparison_markdown": None,
        "comparison_report": None,
        "engineer_report": None,
        "full_markdown_report": None,
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "ingestion",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
        "chunk_indexing_status": None,
        "chunk_indexing_error": None,
        "chunk_count": None,
        "messages": [],
        "errors": [],
        "agent_runs": [],
        "cost_tracking": {},
    }
