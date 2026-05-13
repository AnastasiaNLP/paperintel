import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from agents.llm_provider import call_text_llm
from api.session_store import SessionStore
from config.settings import settings
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.errors import make_error
from models.qa import Intent, IntentResolution
from models.session import Session, Turn

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "intent_router.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

RECENT_TURNS_LIMIT = 10
MAX_TURN_CHARS = 500
QA_INTENTS = {"qa_factual", "qa_math", "qa_comparison", "qa_followup"}


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _agent_run_persistence(config: RunnableConfig | None) -> AgentRunPersistence:
    configurable = _configurable(config)
    persistence = configurable.get("agent_run_persistence")
    if persistence is None:
        persistence = (
            config.get("agent_run_persistence") if isinstance(config, dict) else None
        )
    if persistence is not None and hasattr(persistence, "save"):
        return persistence
    return NoopAgentRunPersistence()


def _session_store(config: RunnableConfig | None) -> SessionStore | None:
    store = _configurable(config).get("session_store")
    if store is not None and hasattr(store, "require_session"):
        return store
    return None


def _policy_snapshot(policy: AgentRuntimePolicy) -> dict[str, Any]:
    return policy.model_dump(mode="json")


def _apply_policy_warning(run: AgentRun, policy: AgentRuntimePolicy) -> None:
    if run.llm_call_count <= policy.max_tool_calls:
        return
    run.details["policy_warning"] = "exceeded_max_tool_calls"
    run.details["policy_max_tool_calls"] = policy.max_tool_calls
    run.details["actual_llm_call_count"] = run.llm_call_count


def _start_router_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    session_id = configurable.get("session_id") or state.get("session_id")
    return AgentRun(
        agent_name="intent_router",
        session_id=session_id,
        job_id=configurable.get("job_id"),
        input_refs=[
            "state:user_message",
            f"session:{session_id}" if session_id else "session:<missing>",
            f"turns:recent:{RECENT_TURNS_LIMIT}",
        ],
        model=settings.haiku_model,
        iteration_count=1,
    )


def _with_agent_run(
    result: dict,
    run: AgentRun,
    persistence: AgentRunPersistence,
) -> dict:
    persistence.save(run)
    result["agent_runs"] = [run]
    return result


def _structured_error(
    message: str,
    *,
    run: AgentRun,
    session_id: str | None,
    stage: str,
) -> Any:
    return make_error(
        "INTENT_ROUTER_FAILED",
        message,
        node="intent_router",
        agent="intent_router",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _format_turn(turn: Turn) -> dict[str, Any]:
    content = turn.content.strip()
    if len(content) > MAX_TURN_CHARS:
        content = f"{content[:MAX_TURN_CHARS]}..."
    return {
        "role": turn.role,
        "content": content,
        "intent": turn.intent,
        "referenced_paper_ids": turn.referenced_paper_ids,
    }


def _build_router_json(
    *,
    message: str,
    session: Session,
    recent_turns: list[Turn],
) -> str:
    payload = {
        "message": message,
        "persona": session.persona,
        "active_paper_ids": session.active_paper_ids,
        "recent_turns": [_format_turn(turn) for turn in recent_turns],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_resolution_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Intent Router JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected intent resolution object, got {type(data).__name__}"

    return data, None


def _clarification_resolution(
    question: str,
    *,
    confidence: float = 0.0,
    reasoning: str | None = None,
) -> IntentResolution:
    return IntentResolution(
        intent="clarification_needed",
        referenced_paper_ids=[],
        ambiguous=True,
        clarification_question=question,
        confidence=confidence,
        reasoning=reasoning,
    )


def _normalize_resolution(
    payload: dict[str, Any],
    *,
    active_paper_ids: list[str],
) -> tuple[IntentResolution | None, str | None]:
    try:
        resolution = IntentResolution(
            intent=payload.get("intent"),
            referenced_paper_ids=payload.get("referenced_paper_ids", []) or [],
            ambiguous=bool(payload.get("ambiguous", False)),
            clarification_question=payload.get("clarification_question"),
            confidence=payload.get("confidence", 1.0),
            reasoning=payload.get("reasoning"),
        )
    except ValidationError as exc:
        return None, f"Intent resolution normalization failed: {exc}"

    known = set(active_paper_ids)
    unknown = [
        paper_id
        for paper_id in resolution.referenced_paper_ids
        if paper_id not in known
    ]
    if unknown:
        return (
            _clarification_resolution(
                (
                    "Please clarify which analyzed paper you mean. "
                    f"Available paper ids: {', '.join(active_paper_ids) or 'none'}."
                ),
                confidence=min(resolution.confidence, 0.3),
                reasoning=(
                    "Router returned referenced paper ids outside active_paper_ids: "
                    f"{', '.join(unknown)}"
                ),
            ),
            None,
        )

    if resolution.intent in QA_INTENTS and not active_paper_ids:
        return (
            _clarification_resolution(
                (
                    "I do not have an analyzed paper in this session yet. "
                    "Send a paper URL first, or specify which paper to analyze."
                ),
                confidence=min(resolution.confidence, 0.3),
                reasoning="QA intent requires at least one active analyzed paper.",
            ),
            None,
        )

    return resolution, None


def _call_llm(
    user_content: str,
    *,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    return call_text_llm(
        requested_model=settings.haiku_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
        context_label="Intent Router",
    )


def _result_from_resolution(
    resolution: IntentResolution,
    *,
    persona: str,
) -> dict[str, Any]:
    return {
        "intent_resolution": resolution,
        "intent": resolution.intent,
        "referenced_paper_ids": resolution.referenced_paper_ids,
        "persona": persona,
        "needs_clarification": resolution.ambiguous,
        "clarification_question": resolution.clarification_question,
    }


def intent_router_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Intent Router started")
    run = _start_router_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("intent_router", config)
    run.details["policy_applied"] = _policy_snapshot(policy)
    session_id = run.session_id

    message = str(state.get("user_message") or state.get("message") or "").strip()
    if not session_id:
        error = "Intent Router requires session_id"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="input",
                    )
                ]
            },
            run,
            persistence,
        )
    if not message:
        error = "Intent Router requires user_message"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="input",
                    )
                ]
            },
            run,
            persistence,
        )

    store = _session_store(config)
    if store is None:
        error = "Intent Router requires session_store in config"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "session_context"},
        )
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="session_context",
                    )
                ]
            },
            run,
            persistence,
        )

    try:
        session = store.require_session(session_id)
        recent_turns = store.list_recent_turns(session_id, limit=RECENT_TURNS_LIMIT)
    except Exception as exc:
        error = f"Intent Router failed to load session context: {exc}"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "session_context"},
        )
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="session_context",
                    )
                ]
            },
            run,
            persistence,
        )

    run.llm_call_count += 1
    raw, llm_error = _call_llm(
        _build_router_json(
            message=message,
            session=session,
            recent_turns=recent_turns,
        ),
        max_tokens=policy.max_tokens or 1500,
    )
    if llm_error:
        resolution = _clarification_resolution(
            "I could not confidently route this message. Please clarify what you want to do.",
            confidence=0.0,
            reasoning=llm_error,
        )
        run.fallback(
            output_ref="state:intent_resolution",
            details={
                "fallback_used": True,
                "fallback_reason": "llm_error",
                "error": llm_error,
            },
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            _result_from_resolution(resolution, persona=session.persona),
            run,
            persistence,
        )

    payload, parse_error = _parse_resolution_payload(raw or "")
    if parse_error or payload is None:
        error = parse_error or "Intent Router parse failed"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "parse"})
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="parse",
                    )
                ]
            },
            run,
            persistence,
        )

    resolution, normalization_error = _normalize_resolution(
        payload,
        active_paper_ids=session.active_paper_ids,
    )
    if normalization_error or resolution is None:
        error = normalization_error or "Intent Router normalization failed"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "normalization"},
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="normalization",
                    )
                ]
            },
            run,
            persistence,
        )

    run.complete(
        output_ref="state:intent_resolution",
        confidence=resolution.confidence,
        details={
            "intent": resolution.intent,
            "ambiguous": resolution.ambiguous,
            "referenced_paper_count": len(resolution.referenced_paper_ids),
            "active_paper_count": len(session.active_paper_ids),
            "persona": session.persona,
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        _result_from_resolution(resolution, persona=session.persona),
        run,
        persistence,
    )
