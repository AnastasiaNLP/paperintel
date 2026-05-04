from typing import Any, Protocol

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from api.session_store import SessionStore
from models.errors import ErrorCodes, StructuredError, make_error
from models.session import GraphInvocationResult, HandlerResult, Persona, Session


class ConversationRunner(Protocol):
    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        ...


class ChatHandler:
    def __init__(
        self,
        *,
        store: SessionStore,
        conversation_runner: ConversationRunner,
        agent_run_persistence: AgentRunPersistence | None = None,
    ) -> None:
        self.store = store
        self.conversation_runner = conversation_runner
        self.agent_run_persistence = (
            agent_run_persistence or NoopAgentRunPersistence()
        )

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
            graph_result = self._invoke_graph(session, message)
        except Exception as exc:
            error = make_error(
                ErrorCodes.FATAL_ERROR,
                f"Conversation graph failed: {exc}",
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
                content="I could not complete the request because the conversation graph failed.",
                error=error,
            )
            return HandlerResult(
                session_id=session.id,
                response_text=assistant_turn.content,
                phase="failed",
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
            artifact_refs=graph_result.artifact_refs,
            user_turn_id=user_turn.id,
            assistant_turn_id=assistant_turn.id,
        )

    def _invoke_graph(self, session: Session, message: str) -> GraphInvocationResult:
        raw = self.conversation_runner.invoke(
            {
                "session_id": session.id,
                "message": message,
                "phase": session.phase,
                "persona": session.persona,
            },
            config={
                "configurable": {
                    "session_id": session.id,
                    "agent_run_persistence": self.agent_run_persistence,
                }
            },
        )
        return _normalize_graph_result(raw)


def _normalize_graph_result(raw: dict[str, Any]) -> GraphInvocationResult:
    return GraphInvocationResult(
        response_text=str(raw.get("response_text") or ""),
        intent=raw.get("intent"),
        referenced_paper_ids=list(raw.get("referenced_paper_ids") or []),
        artifact_refs=list(raw.get("artifact_refs") or []),
        next_phase=raw.get("next_phase"),
        raw=raw,
    )
