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
from models.errors import make_error
from models.qa import EvidencePlan, IntentResolution
from models.retrieval import ChunkSearchQuery, ChunkType, EvidenceBundle
from services.retrieval_layer import RetrievalLayer

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "retrieval_planner.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

QA_INTENTS = {"qa_factual", "qa_math", "qa_comparison", "qa_followup"}
DEFAULT_K = 8
REPLAN_K = 12


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


def _retrieval_layer(config: RunnableConfig | None) -> RetrievalLayer | None:
    layer = _configurable(config).get("retrieval_layer")
    if layer is not None and hasattr(layer, "search_chunks"):
        return layer
    return None


def _policy_snapshot(policy: AgentRuntimePolicy) -> dict[str, Any]:
    return policy.model_dump(mode="json")


def _apply_policy_warning(run: AgentRun, policy: AgentRuntimePolicy) -> None:
    if run.llm_call_count <= policy.max_tool_calls:
        return
    run.details["policy_warning"] = "exceeded_max_tool_calls"
    run.details["policy_max_tool_calls"] = policy.max_tool_calls
    run.details["actual_llm_call_count"] = run.llm_call_count


def _start_planner_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    input_refs = ["state:user_message", "state:intent_resolution"]
    if state.get("referenced_paper_ids"):
        input_refs.append("state:referenced_paper_ids")
    return AgentRun(
        agent_name="retrieval_planner",
        session_id=configurable.get("session_id") or state.get("session_id"),
        job_id=configurable.get("job_id"),
        input_refs=input_refs,
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
        "RETRIEVAL_PLANNER_FAILED",
        message,
        node="retrieval_planner",
        agent="retrieval_planner",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _persona_chunk_priorities(persona: str, intent: str) -> list[ChunkType]:
    if intent == "qa_math":
        return ["equation", "text"]
    if intent == "qa_comparison":
        return ["table", "text"] if persona == "techlead" else ["text", "table"]
    if persona == "engineer":
        return ["text", "table"]
    if persona == "researcher":
        return ["text", "equation"]
    if persona == "techlead":
        return ["table", "text"]
    return ["text"]


def _section_queries(persona: str, intent: str) -> list[str]:
    if intent == "qa_math":
        return ["Method", "Objective", "Loss", "Training"]
    if intent == "qa_comparison":
        return ["Results", "Experiments", "Limitations", "Discussion"]
    if persona == "engineer":
        return ["Method", "Implementation", "Results"]
    if persona == "researcher":
        return ["Method", "Related Work", "Limitations", "Discussion"]
    if persona == "techlead":
        return ["Results", "Limitations", "Deployment"]
    return ["Method", "Results"]


def _query_with_section_hints(question: str, sections: list[str]) -> str:
    parts = [question.strip(), *sections]
    return " ".join(part for part in parts if part)


def _initial_plan(
    *,
    question: str,
    intent: str,
    persona: str,
    paper_ids: list[str],
) -> EvidencePlan:
    sections = _section_queries(persona, intent)
    return EvidencePlan(
        intent=intent,
        paper_ids=paper_ids,
        search_query=_query_with_section_hints(question, sections),
        chunk_types_priority=_persona_chunk_priorities(persona, intent),
        section_queries=sections,
        k=DEFAULT_K,
    )


def _search_query_from_plan(
    *,
    plan: EvidencePlan,
    session_id: str | None,
) -> ChunkSearchQuery:
    return ChunkSearchQuery(
        query=plan.search_query,
        session_id=session_id,
        paper_ids=plan.paper_ids,
        limit=plan.k,
        filters={"chunk_type": plan.chunk_types_priority},
    )


def _retrieve(
    *,
    layer: RetrievalLayer,
    plan: EvidencePlan,
    session_id: str | None,
) -> EvidenceBundle:
    query = _search_query_from_plan(plan=plan, session_id=session_id)
    results = layer.search_chunks(query)
    return layer.assemble_evidence(plan.search_query, results, max_chunks=plan.k)


def _parse_replan_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Retrieval Planner JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected replan object, got {type(data).__name__}"

    return data, None


def _normalize_replan(
    payload: dict[str, Any],
    *,
    failed_plan: EvidencePlan,
) -> tuple[EvidencePlan | None, str | None]:
    chunk_types = payload.get("chunk_types_priority", failed_plan.chunk_types_priority)
    section_queries = payload.get("section_queries", failed_plan.section_queries)
    k = payload.get("k", REPLAN_K)
    search_query = payload.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        return None, "search_query must be a non-empty string"
    if not isinstance(chunk_types, list):
        return None, "chunk_types_priority must be a list"
    if not isinstance(section_queries, list):
        return None, "section_queries must be a list"

    try:
        return (
            EvidencePlan(
                intent=failed_plan.intent,
                paper_ids=failed_plan.paper_ids,
                search_query=search_query,
                chunk_types_priority=chunk_types,
                section_queries=section_queries,
                k=k,
                requires_replanning=True,
                replanning_reason=payload.get("replanning_reason")
                or "Initial retrieval returned no evidence.",
                fallback_used=True,
            ),
            None,
        )
    except ValidationError as exc:
        return None, f"Replan normalization failed: {exc}"


def _build_replan_json(
    *,
    question: str,
    persona: str,
    failed_plan: EvidencePlan,
    evidence: EvidenceBundle,
) -> str:
    payload = {
        "question": question,
        "persona": persona,
        "failed_plan": failed_plan.model_dump(mode="json"),
        "coverage_notes": evidence.coverage_notes,
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
        context_label="Retrieval Planner",
    )


def _resolution_from_state(state: dict[str, Any]) -> IntentResolution | None:
    resolution = state.get("intent_resolution")
    return resolution if isinstance(resolution, IntentResolution) else None


def _referenced_paper_ids(state: dict[str, Any], resolution: IntentResolution) -> list[str]:
    paper_ids = state.get("referenced_paper_ids") or resolution.referenced_paper_ids
    return list(paper_ids or [])


def retrieval_planner_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Retrieval Planner started")
    run = _start_planner_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("retrieval_planner", config)
    run.details["policy_applied"] = _policy_snapshot(policy)
    session_id = run.session_id

    question = str(state.get("user_message") or state.get("message") or "").strip()
    persona = str(state.get("persona") or "").strip()
    resolution = _resolution_from_state(state)
    layer = _retrieval_layer(config)

    if not question:
        error = "Retrieval Planner requires user_message"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {"errors": [_structured_error(error, run=run, session_id=session_id, stage="input")]},
            run,
            persistence,
        )
    if not persona:
        error = "Retrieval Planner requires persona"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {"errors": [_structured_error(error, run=run, session_id=session_id, stage="input")]},
            run,
            persistence,
        )
    if resolution is None:
        error = "Retrieval Planner requires intent_resolution"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {"errors": [_structured_error(error, run=run, session_id=session_id, stage="input")]},
            run,
            persistence,
        )
    if resolution.intent not in QA_INTENTS:
        error = f"Retrieval Planner only supports QA intents, got {resolution.intent}"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "intent"})
        return _with_agent_run(
            {"errors": [_structured_error(error, run=run, session_id=session_id, stage="intent")]},
            run,
            persistence,
        )
    if layer is None:
        error = "Retrieval Planner requires retrieval_layer in config"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "retrieval_layer"},
        )
        return _with_agent_run(
            {
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="retrieval_layer",
                    )
                ]
            },
            run,
            persistence,
        )

    paper_ids = _referenced_paper_ids(state, resolution)
    if not paper_ids:
        error = "Retrieval Planner requires referenced_paper_ids for QA"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {"errors": [_structured_error(error, run=run, session_id=session_id, stage="input")]},
            run,
            persistence,
        )

    plan = _initial_plan(
        question=question,
        intent=resolution.intent,
        persona=persona,
        paper_ids=paper_ids,
    )
    evidence = _retrieve(layer=layer, plan=plan, session_id=session_id)
    iterations_used = 1

    if not evidence.results and policy.max_iterations > 1:
        run.llm_call_count += 1
        raw, llm_error = _call_llm(
            _build_replan_json(
                question=question,
                persona=persona,
                failed_plan=plan,
                evidence=evidence,
            ),
            max_tokens=policy.max_tokens or 1600,
        )
        if llm_error:
            plan = plan.model_copy(
                update={
                    "requires_replanning": True,
                    "replanning_reason": llm_error,
                    "fallback_used": True,
                }
            )
        else:
            payload, parse_error = _parse_replan_payload(raw or "")
            if parse_error or payload is None:
                plan = plan.model_copy(
                    update={
                        "requires_replanning": True,
                        "replanning_reason": parse_error or "Replan parse failed",
                        "fallback_used": True,
                    }
                )
            else:
                replanned, normalization_error = _normalize_replan(
                    payload,
                    failed_plan=plan,
                )
                if replanned is None:
                    plan = plan.model_copy(
                        update={
                            "requires_replanning": True,
                            "replanning_reason": normalization_error
                            or "Replan normalization failed",
                            "fallback_used": True,
                        }
                    )
                else:
                    plan = replanned
                    evidence = _retrieve(layer=layer, plan=plan, session_id=session_id)
                    iterations_used = 2

    if not evidence.results and not plan.fallback_used:
        plan = plan.model_copy(update={"fallback_used": True})

    run.complete(
        output_ref="state:evidence_bundle",
        details={
            "intent": resolution.intent,
            "persona": persona,
            "paper_count": len(paper_ids),
            "evidence_count": len(evidence.results),
            "iterations_used": iterations_used,
            "fallback_used": plan.fallback_used,
            "requires_replanning": plan.requires_replanning,
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {
            "evidence_plan": plan,
            "evidence_bundle": evidence,
        },
        run,
        persistence,
    )
