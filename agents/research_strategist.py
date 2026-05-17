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
from models.discovery import DiscoveryPlan, ResearchQuery
from models.errors import make_error

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "research_strategist.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

MAX_QUERIES = 4
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_PER_QUERY = 10


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


def _start_strategist_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    session_id = configurable.get("session_id") or state.get("session_id")
    return AgentRun(
        agent_name="research_strategist",
        session_id=session_id,
        job_id=configurable.get("job_id"),
        input_refs=[
            "state:user_message",
            "state:persona",
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
        "RESEARCH_STRATEGIST_FAILED",
        message,
        node="research_strategist",
        agent="research_strategist",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _parse_plan_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Research Strategist JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected discovery plan object, got {type(data).__name__}"

    return data, None


def _coerce_max_results(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS
    return max(1, min(parsed, MAX_RESULTS_PER_QUERY))


def _normalize_query_payload(item: Any) -> ResearchQuery | None:
    if isinstance(item, str):
        query = item.strip()
        max_results = DEFAULT_MAX_RESULTS
        source = "arxiv"
    elif isinstance(item, dict):
        query = str(item.get("query") or "").strip()
        max_results = _coerce_max_results(item.get("max_results", DEFAULT_MAX_RESULTS))
        source = str(item.get("source") or "arxiv").strip().lower()
    else:
        return None

    if not query:
        return None
    if source != "arxiv":
        source = "arxiv"

    try:
        return ResearchQuery(
            query=query,
            max_results=max_results,
            source=source,
        )
    except ValidationError:
        return None


def _normalize_plan(
    payload: dict[str, Any],
    *,
    fallback_topic: str,
) -> tuple[DiscoveryPlan | None, str | None]:
    topic = str(payload.get("topic") or "").strip() or fallback_topic
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        return None, "queries must be a list"

    queries: list[ResearchQuery] = []
    seen: set[str] = set()
    for item in raw_queries:
        query = _normalize_query_payload(item)
        if query is None:
            continue
        key = query.query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= MAX_QUERIES:
            break

    if not queries:
        return None, "Discovery plan must include at least one valid query"

    try:
        return DiscoveryPlan(topic=topic, queries=queries), None
    except ValidationError as exc:
        return None, f"Discovery plan normalization failed: {exc}"


def _fallback_plan(topic: str) -> DiscoveryPlan:
    return DiscoveryPlan(
        topic=topic,
        queries=[
            ResearchQuery(
                query=topic,
                max_results=DEFAULT_MAX_RESULTS,
                source="arxiv",
            )
        ],
    )


def _build_strategy_json(
    *,
    message: str,
    persona: str,
) -> str:
    payload = {
        "message": message,
        "persona": persona,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
        context_label="Research Strategist",
    )


def _fallback_result(
    *,
    topic: str,
    run: AgentRun,
    persistence: AgentRunPersistence,
    policy: AgentRuntimePolicy,
    reason: str,
) -> dict:
    plan = _fallback_plan(topic)
    run.fallback(
        output_ref="state:discovery_plan",
        details={
            "fallback_used": True,
            "fallback_reason": reason,
            "topic": plan.topic,
            "query_count": len(plan.queries),
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {
            "discovery_topic": plan.topic,
            "discovery_plan": plan,
        },
        run,
        persistence,
    )


def research_strategist_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Research Strategist started")
    run = _start_strategist_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("research_strategist", config)
    run.details["policy_applied"] = _policy_snapshot(policy)
    session_id = run.session_id

    message = str(state.get("user_message") or state.get("message") or "").strip()
    persona = str(state.get("persona") or "engineer").strip() or "engineer"
    if not message:
        error = "Research Strategist requires user_message"
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

    run.llm_call_count += 1
    raw, llm_error = _call_llm(
        _build_strategy_json(message=message, persona=persona),
        max_tokens=policy.max_tokens or 1500,
    )
    if llm_error:
        return _fallback_result(
            topic=message,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=llm_error,
        )

    payload, parse_error = _parse_plan_payload(raw or "")
    if parse_error or payload is None:
        return _fallback_result(
            topic=message,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=parse_error or "Research Strategist parse failed",
        )

    plan, normalization_error = _normalize_plan(payload, fallback_topic=message)
    if normalization_error or plan is None:
        return _fallback_result(
            topic=message,
            run=run,
            persistence=persistence,
            policy=policy,
            reason=normalization_error or "Research Strategist normalization failed",
        )

    run.complete(
        output_ref="state:discovery_plan",
        details={
            "topic": plan.topic,
            "query_count": len(plan.queries),
            "persona": persona,
            "fallback_used": False,
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {
            "discovery_topic": plan.topic,
            "discovery_plan": plan,
        },
        run,
        persistence,
    )
