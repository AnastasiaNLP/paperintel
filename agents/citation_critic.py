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
from models.qa import AnswerDraft, CriticReview
from models.retrieval import EvidenceBundle
from services.repair import (
    build_repair_context,
    is_repair_exhausted,
    latest_agent_run_id,
    normalize_repair_decision,
    should_trigger_repair,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "citation_critic.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

MAX_EVIDENCE_CHARS_PER_CHUNK = 1800


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


def _start_critic_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    input_refs = ["state:answer_draft"]
    if state.get("evidence_bundle") is not None:
        input_refs.append("state:evidence_bundle")

    answer_run_id = latest_agent_run_id(
        state.get("agent_runs", []) or [],
        agent_name="answer_agent",
    )
    if answer_run_id:
        input_refs.append(answer_run_id)

    return AgentRun(
        agent_name="citation_critic",
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
        "CITATION_CRITIC_FAILED",
        message,
        node="citation_critic",
        agent="citation_critic",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _build_review_json(
    answer_draft: AnswerDraft,
    evidence: EvidenceBundle,
) -> str:
    answer_citation_ids = [citation.chunk_id for citation in answer_draft.citations]
    payload = {
        "answer": {
            "id": answer_draft.id,
            "question": answer_draft.question,
            "answer_text": answer_draft.answer_text,
            "citation_chunk_ids": answer_citation_ids,
            "confidence": answer_draft.confidence,
            "repair_iteration": answer_draft.repair_iteration,
        },
        "evidence": {
            "query": evidence.query,
            "coverage_notes": evidence.coverage_notes,
            "chunks": [
                {
                    "chunk_id": result.chunk.id,
                    "paper_id": result.chunk.paper_id,
                    "score": result.score,
                    "rank": result.rank,
                    "chunk_type": result.chunk.chunk_type,
                    "page_start": result.chunk.location.page_start,
                    "page_end": result.chunk.location.page_end,
                    "section_title": result.chunk.location.section_title,
                    "text": result.chunk.text[:MAX_EVIDENCE_CHARS_PER_CHUNK],
                }
                for result in evidence.results
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_review_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Citation Critic JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected critic review object, got {type(data).__name__}"

    return data, None


def _normalize_string_list(value: object, field_name: str) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [], f"{field_name} must be a list of strings"
    return [item.strip() for item in value if item.strip()], None


def _normalize_review(
    payload: dict[str, Any],
    *,
    answer_draft: AnswerDraft,
) -> tuple[CriticReview | None, str | None]:
    unsupported_claims, error = _normalize_string_list(
        payload.get("unsupported_claims", []),
        "unsupported_claims",
    )
    if error:
        return None, error

    missing_evidence, error = _normalize_string_list(
        payload.get("missing_evidence", []),
        "missing_evidence",
    )
    if error:
        return None, error

    contradictions, error = _normalize_string_list(
        payload.get("contradictions", []),
        "contradictions",
    )
    if error:
        return None, error

    repair_instructions, error = _normalize_string_list(
        payload.get("repair_instructions", []),
        "repair_instructions",
    )
    if error:
        return None, error

    confidence_adjustments = payload.get("confidence_adjustments", {})
    if not isinstance(confidence_adjustments, dict):
        return None, "confidence_adjustments must be an object"

    needs_repair, repair_instructions = normalize_repair_decision(
        needs_repair=bool(payload.get("needs_repair", False)),
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
        contradictions=contradictions,
        repair_instructions=repair_instructions,
    )

    try:
        return (
            CriticReview(
                reviewed_answer_id=answer_draft.id,
                unsupported_claims=unsupported_claims,
                missing_evidence=missing_evidence,
                contradictions=contradictions,
                confidence_adjustments=confidence_adjustments,
                needs_repair=needs_repair,
                repair_target_agent="answer_agent" if needs_repair else None,
                repair_instructions=repair_instructions,
                critic_confidence=payload.get("critic_confidence", 1.0),
            ),
            None,
        )
    except ValidationError as exc:
        return None, f"Critic review normalization failed: {exc}"


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
        context_label="Citation Critic",
    )


def _downgrade_answer(answer_draft: AnswerDraft, review: CriticReview) -> AnswerDraft:
    notes = []
    if review.unsupported_claims:
        notes.append("unsupported claims")
    if review.missing_evidence:
        notes.append("missing evidence")
    if review.contradictions:
        notes.append("contradictions")
    reason = ", ".join(notes) or "citation review concerns"
    return answer_draft.model_copy(
        update={
            "answer_text": (
                f"{answer_draft.answer_text}\n\n"
                f"Limited evidence note: Citation Critic flagged {reason}. "
                "Treat the answer as provisional."
            ),
            "confidence": min(answer_draft.confidence, 0.3),
            "limitations_noted": True,
        }
    )


def citation_critic_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Citation Critic started")
    run = _start_critic_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("citation_critic", config)
    run.details["policy_applied"] = _policy_snapshot(policy)
    session_id = run.session_id

    answer_draft = state.get("answer_draft")
    if not isinstance(answer_draft, AnswerDraft):
        error = "Citation Critic requires answer_draft"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {
                "repair_context": None,
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

    if answer_draft.insufficient_evidence:
        review = CriticReview(
            reviewed_answer_id=answer_draft.id,
            critic_confidence=1.0,
        )
        run.complete(
            output_ref="state:critic_review",
            termination_reason="skipped",
            details={
                "fallback_used": True,
                "fallback_reason": "insufficient_evidence_no_review_needed",
            },
        )
        return _with_agent_run(
            {"critic_review": review, "repair_context": None},
            run,
            persistence,
        )

    evidence = state.get("evidence_bundle")
    if not isinstance(evidence, EvidenceBundle) or not evidence.results:
        error = "Citation Critic requires non-empty evidence_bundle"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "input"})
        return _with_agent_run(
            {
                "repair_context": None,
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
        _build_review_json(answer_draft, evidence),
        max_tokens=policy.max_tokens or 1600,
    )
    if llm_error:
        run.fail(
            output_ref="state:errors",
            details={"error": llm_error, "stage": "llm_call"},
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            {
                "repair_context": None,
                "errors": [
                    _structured_error(
                        llm_error,
                        run=run,
                        session_id=session_id,
                        stage="llm_call",
                    )
                ]
            },
            run,
            persistence,
        )

    payload, parse_error = _parse_review_payload(raw or "")
    if parse_error or payload is None:
        error = parse_error or "Citation Critic parse failed"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "parse"},
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            {
                "repair_context": None,
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

    review, normalization_error = _normalize_review(payload, answer_draft=answer_draft)
    if normalization_error or review is None:
        error = normalization_error or "Citation Critic normalization failed"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "normalization"},
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(
            {
                "repair_context": None,
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

    result: dict[str, Any] = {"critic_review": review, "repair_context": None}
    details = {
        "needs_repair": review.needs_repair,
        "unsupported_claims": len(review.unsupported_claims),
        "missing_evidence": len(review.missing_evidence),
        "contradictions": len(review.contradictions),
        "repair_iteration": answer_draft.repair_iteration,
    }

    if should_trigger_repair(review) and is_repair_exhausted(answer_draft):
        downgraded = _downgrade_answer(answer_draft, review)
        result["answer_draft"] = downgraded
        run.fallback(
            output_ref="state:critic_review",
            details={
                **details,
                "fallback_used": True,
                "fallback_reason": "repair_limit_exhausted_downgrade",
            },
        )
        _apply_policy_warning(run, policy)
        return _with_agent_run(result, run, persistence)

    if should_trigger_repair(review):
        original_run_id = latest_agent_run_id(
            state.get("agent_runs", []) or [],
            agent_name="answer_agent",
        ) or answer_draft.id
        result["repair_context"] = build_repair_context(
            review,
            answer_draft,
            original_run_id,
        )
        result["answer_draft"] = None

    run.complete(output_ref="state:critic_review", details=details)
    _apply_policy_warning(run, policy)
    return _with_agent_run(result, run, persistence)
