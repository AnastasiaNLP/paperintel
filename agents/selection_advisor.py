import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.discovery import SearchCandidate, SelectionAdvice
from models.errors import make_error

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "selection_advisor.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

FALLBACK_DISPLAY_LIMIT = 5
FALLBACK_RECOMMENDATION_LIMIT = 3
MAX_CANDIDATE_ABSTRACT_CHARS = 700


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


def _policy_snapshot(policy: AgentRuntimePolicy) -> dict[str, Any]:
    return policy.model_dump(mode="json")


def _apply_policy_warning(run: AgentRun, policy: AgentRuntimePolicy) -> None:
    if run.llm_call_count <= policy.max_tool_calls:
        return
    run.details["policy_warning"] = "exceeded_max_tool_calls"
    run.details["policy_max_tool_calls"] = policy.max_tool_calls
    run.details["actual_llm_call_count"] = run.llm_call_count


def _candidate_refs(candidates: list[SearchCandidate]) -> list[str]:
    return [f"search_candidate:{candidate.id}" for candidate in candidates]


def _start_advisor_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    return AgentRun(
        agent_name="selection_advisor",
        session_id=configurable.get("session_id") or state.get("session_id"),
        job_id=configurable.get("job_id"),
        input_refs=[
            "state:discovery_topic",
            "state:search_candidates",
            *_candidate_refs(_candidates_from_state(state)),
        ],
        model=settings.sonnet_model,
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
        "SELECTION_ADVISOR_FAILED",
        message,
        node="selection_advisor",
        agent="selection_advisor",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _parse_advice_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Selection Advisor JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected selection advice object, got {type(data).__name__}"

    return data, None


def _candidates_from_state(state: dict[str, Any]) -> list[SearchCandidate]:
    raw_candidates = state.get("search_candidates") or []
    return [
        candidate
        for candidate in raw_candidates
        if isinstance(candidate, SearchCandidate)
    ]


def _candidate_label(candidate: SearchCandidate) -> str:
    parts = [str(candidate.display_rank), candidate.title]
    if candidate.year:
        parts.append(str(candidate.year))
    if candidate.arxiv_id:
        parts.append(f"arXiv:{candidate.arxiv_id}")
    return " | ".join(parts)


def _build_advisor_json(
    *,
    topic: str,
    persona: str,
    candidates: list[SearchCandidate],
) -> str:
    payload = {
        "topic": topic,
        "persona": persona,
        "candidates": [
            {
                "id": candidate.id,
                "display_rank": candidate.display_rank,
                "title": candidate.title,
                "authors": candidate.authors,
                "year": candidate.year,
                "arxiv_id": candidate.arxiv_id,
                "url": candidate.url,
                "score": candidate.score,
                "reasons": candidate.reasons,
                "abstract": (candidate.abstract or "")[:MAX_CANDIDATE_ABSTRACT_CHARS],
            }
            for candidate in candidates
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _call_llm(
    user_content: str,
    *,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    return call_text_llm(
        requested_model=settings.sonnet_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
        context_label="Selection Advisor",
    )


def _normalize_recommendations(
    value: Any,
    *,
    valid_ids: set[str],
) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        candidate_id = item.strip()
        if candidate_id in valid_ids and candidate_id not in seen:
            normalized.append(candidate_id)
            seen.add(candidate_id)
    return normalized


def _fallback_recommended_ids(candidates: list[SearchCandidate]) -> list[str]:
    return [
        candidate.id
        for candidate in sorted(candidates, key=lambda candidate: candidate.display_rank)[
            :FALLBACK_RECOMMENDATION_LIMIT
        ]
    ]


def _fallback_text(topic: str, candidates: list[SearchCandidate]) -> str:
    if not candidates:
        return (
            f"I did not find candidate papers for '{topic}'. Try a more specific "
            "topic or different keywords."
        )

    lines = [
        f"I found {len(candidates)} candidate papers for '{topic}'.",
        "",
        "Recommended shortlist:",
    ]
    for candidate in sorted(candidates, key=lambda item: item.display_rank)[
        :FALLBACK_DISPLAY_LIMIT
    ]:
        lines.append(f"{candidate.display_rank}. {_candidate_label(candidate)}")
        if candidate.reasons:
            lines.append(f"   Why: {', '.join(candidate.reasons[:2])}")
    lines.extend(
        [
            "",
            "Reply with the numbers you want to analyze, for example: 1, 3, 5.",
        ]
    )
    return "\n".join(lines)


def _fallback_advice(topic: str, candidates: list[SearchCandidate]) -> SelectionAdvice:
    return SelectionAdvice(
        topic=topic,
        response_text=_fallback_text(topic, candidates),
        recommended_candidate_ids=_fallback_recommended_ids(candidates),
        candidate_count=len(candidates),
    )


def _normalize_advice(
    payload: dict[str, Any],
    *,
    topic: str,
    candidates: list[SearchCandidate],
) -> tuple[SelectionAdvice | None, str | None]:
    response_text = payload.get("response_text")
    if not isinstance(response_text, str) or not response_text.strip():
        return None, "response_text must be a non-empty string"

    valid_ids = {candidate.id for candidate in candidates}
    recommended_ids = _normalize_recommendations(
        payload.get("recommended_candidate_ids", []),
        valid_ids=valid_ids,
    )
    if not recommended_ids:
        recommended_ids = _fallback_recommended_ids(candidates)

    try:
        return (
            SelectionAdvice(
                topic=topic,
                response_text=response_text,
                recommended_candidate_ids=recommended_ids,
                candidate_count=len(candidates),
            ),
            None,
        )
    except ValidationError as exc:
        return None, f"Selection advice normalization failed: {exc}"


def _fallback_result(
    *,
    topic: str,
    candidates: list[SearchCandidate],
    run: AgentRun,
    persistence: AgentRunPersistence,
    policy: AgentRuntimePolicy,
    reason: str,
) -> dict:
    advice = _fallback_advice(topic, candidates)
    run.fallback(
        output_ref="state:selection_advice",
        details={
            "fallback_used": True,
            "fallback_reason": reason,
            "topic": advice.topic,
            "candidate_count": advice.candidate_count,
            "recommended_count": len(advice.recommended_candidate_ids),
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {
            "selection_advice": advice,
            "response_text": advice.response_text,
        },
        run,
        persistence,
    )


def selection_advisor_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Selection Advisor started")
    run = _start_advisor_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("selection_advisor", config)
    run.details["policy_applied"] = _policy_snapshot(policy)
    session_id = run.session_id

    topic = str(state.get("discovery_topic") or state.get("user_message") or "").strip()
    persona = str(state.get("persona") or "engineer").strip() or "engineer"
    candidates = sorted(
        _candidates_from_state(state),
        key=lambda candidate: candidate.display_rank,
    )

    if not topic:
        error = "Selection Advisor requires discovery_topic"
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

    if not candidates:
        return _fallback_result(
            topic=topic,
            candidates=candidates,
            run=run,
            persistence=persistence,
            policy=policy,
            reason="no_candidates_available",
        )

    run.llm_call_count += 1
    raw, llm_error = _call_llm(
        _build_advisor_json(topic=topic, persona=persona, candidates=candidates),
        max_tokens=policy.max_tokens or 2500,
    )
    if llm_error:
        return _fallback_result(
            topic=topic,
            candidates=candidates,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=llm_error,
        )

    payload, parse_error = _parse_advice_payload(raw or "")
    if parse_error or payload is None:
        return _fallback_result(
            topic=topic,
            candidates=candidates,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=parse_error or "Selection Advisor parse failed",
        )

    advice, normalization_error = _normalize_advice(
        payload,
        topic=topic,
        candidates=candidates,
    )
    if normalization_error or advice is None:
        return _fallback_result(
            topic=topic,
            candidates=candidates,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=normalization_error or "Selection Advisor normalization failed",
        )

    run.complete(
        output_ref="state:selection_advice",
        details={
            "topic": advice.topic,
            "candidate_count": advice.candidate_count,
            "recommended_count": len(advice.recommended_candidate_ids),
            "persona": persona,
            "fallback_used": False,
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {
            "selection_advice": advice,
            "response_text": advice.response_text,
        },
        run,
        persistence,
    )
