import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import TypeAdapter, ValidationError

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.errors import make_error
from models.qa import AnswerDraft, RepairContext
from models.retrieval import CitationRef, EvidenceBundle
from models.session import Persona

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "config" / "prompts"
_PROMPT_PATHS = {
    "engineer": _PROMPT_DIR / "answer_engineer.txt",
    "researcher": _PROMPT_DIR / "answer_researcher.txt",
    "techlead": _PROMPT_DIR / "answer_techlead.txt",
}
_PROMPTS = {
    persona: path.read_text(encoding="utf-8")
    for persona, path in _PROMPT_PATHS.items()
}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_PERSONA_ADAPTER = TypeAdapter(Persona)

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


def _start_answer_run(
    state: dict[str, Any],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    input_refs = ["state:user_message"]
    if state.get("evidence_bundle") is not None:
        input_refs.append("state:evidence_bundle")
    repair_context = state.get("repair_context")
    if isinstance(repair_context, RepairContext):
        input_refs.append(f"repair_context:{repair_context.id}")

    return AgentRun(
        agent_name="answer_agent",
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
        "ANSWER_AGENT_FAILED",
        message,
        node="answer_agent",
        agent="answer_agent",
        severity="error",
        recoverable=True,
        session_id=session_id,
        agent_run_id=run.id,
        stage=stage,
    )


def _resolve_persona(value: object) -> tuple[Persona | None, str | None]:
    try:
        return _PERSONA_ADAPTER.validate_python(value), None
    except ValidationError as exc:
        return None, f"Answer Agent requires valid persona: {exc.errors()[0]['msg']}"


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _citation_map(evidence: EvidenceBundle) -> dict[str, CitationRef]:
    citations_by_chunk_id = {citation.chunk_id: citation for citation in evidence.citations}
    for result in evidence.results:
        chunk = result.chunk
        citations_by_chunk_id.setdefault(
            chunk.id,
            CitationRef(
                paper_id=chunk.paper_id,
                chunk_id=chunk.id,
                page_start=chunk.location.page_start,
                page_end=chunk.location.page_end,
                section_title=chunk.location.section_title,
                artifact_refs=chunk.artifact_refs,
            ),
        )
    return citations_by_chunk_id


def _build_evidence_json(question: str, evidence: EvidenceBundle) -> str:
    payload = {
        "question": question,
        "retrieval_query": evidence.query,
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
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_answer_payload(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Answer JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected answer object, got {type(data).__name__}"

    return data, None


def _normalize_answer(
    payload: dict[str, Any],
    *,
    question: str,
    persona: Persona,
    evidence: EvidenceBundle,
    repair_iteration: int,
) -> tuple[AnswerDraft | None, str | None]:
    answer_text = payload.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None, "answer_text must be a non-empty string"

    citation_chunk_ids = payload.get("citation_chunk_ids", [])
    if not isinstance(citation_chunk_ids, list) or not all(
        isinstance(chunk_id, str) for chunk_id in citation_chunk_ids
    ):
        return None, "citation_chunk_ids must be a list of strings"

    citations_by_chunk_id = _citation_map(evidence)
    unknown = [
        chunk_id for chunk_id in citation_chunk_ids if chunk_id not in citations_by_chunk_id
    ]
    if unknown:
        return None, f"Unknown citation_chunk_ids: {', '.join(sorted(unknown))}"

    confidence = payload.get("confidence", 1.0)
    limitations_noted = payload.get("limitations_noted", False)
    insufficient_evidence = payload.get("insufficient_evidence", False)

    try:
        return (
            AnswerDraft(
                question=question,
                answer_text=answer_text,
                citations=[citations_by_chunk_id[chunk_id] for chunk_id in citation_chunk_ids],
                persona=persona,
                confidence=confidence,
                limitations_noted=bool(limitations_noted),
                insufficient_evidence=bool(insufficient_evidence),
                repair_iteration=repair_iteration,
            ),
            None,
        )
    except ValidationError as exc:
        return None, f"Answer normalization failed: {exc}"


def _call_llm(
    system_prompt: str,
    user_content: str,
    *,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    return call_text_llm(
        requested_model=settings.sonnet_model,
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=max_tokens,
        context_label="Answer Agent",
    )


def _insufficient_evidence_answer(
    *,
    question: str,
    persona: Persona,
    repair_iteration: int,
) -> AnswerDraft:
    return AnswerDraft(
        question=question,
        answer_text=(
            "The indexed evidence is insufficient to answer this question without "
            "guessing."
        ),
        citations=[],
        persona=persona,
        confidence=0.0,
        limitations_noted=True,
        insufficient_evidence=True,
        repair_iteration=repair_iteration,
    )


def answer_agent(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict:
    logger.info("Answer agent started")
    run = _start_answer_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("answer_agent", config)
    run.details["policy_applied"] = _policy_snapshot(policy)

    session_id = run.session_id
    question = str(state.get("user_message") or "").strip()
    persona, persona_error = _resolve_persona(state.get("persona"))
    repair_context = state.get("repair_context")
    repair_iteration = (
        repair_context.iteration if isinstance(repair_context, RepairContext) else 0
    )

    if not question:
        error = "Answer Agent requires user_message"
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

    if persona_error or persona is None:
        error = persona_error or "Answer Agent requires persona"
        run.fail(output_ref="state:errors", details={"error": error, "stage": "persona"})
        return _with_agent_run(
            {
                "repair_context": None,
                "errors": [
                    _structured_error(
                        error,
                        run=run,
                        session_id=session_id,
                        stage="persona",
                    )
                ]
            },
            run,
            persistence,
        )

    evidence = state.get("evidence_bundle")
    if not isinstance(evidence, EvidenceBundle) or not evidence.results:
        answer = _insufficient_evidence_answer(
            question=question,
            persona=persona,
            repair_iteration=repair_iteration,
        )
        run.fallback(
            output_ref="state:answer_draft",
            details={
                "fallback_used": True,
                "fallback_reason": "no_evidence_available",
                "persona": persona,
                "repair_iteration": repair_iteration,
            },
        )
        return _with_agent_run(
            {"answer_draft": answer, "repair_context": None},
            run,
            persistence,
        )

    system_prompt = _PROMPTS[persona]
    user_content = _build_evidence_json(question, evidence)
    if isinstance(repair_context, RepairContext):
        user_content = (
            f"{user_content}\n\n"
            "Repair instructions from Citation Critic:\n"
            f"{json.dumps(repair_context.instructions, ensure_ascii=False, indent=2)}"
        )

    run.llm_call_count += 1
    raw, llm_error = _call_llm(
        system_prompt,
        user_content,
        max_tokens=policy.max_tokens or 1600,
    )
    if llm_error:
        run.fail(
            output_ref="state:errors",
            details={"error": llm_error, "stage": "llm_call", "persona": persona},
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

    payload, parse_error = _parse_answer_payload(raw or "")
    if parse_error or payload is None:
        error = parse_error or "Answer parse failed"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "parse", "persona": persona},
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

    answer, normalization_error = _normalize_answer(
        payload,
        question=question,
        persona=persona,
        evidence=evidence,
        repair_iteration=repair_iteration,
    )
    if normalization_error or answer is None:
        error = normalization_error or "Answer normalization failed"
        run.fail(
            output_ref="state:errors",
            details={"error": error, "stage": "normalization", "persona": persona},
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

    run.complete(
        output_ref="state:answer_draft",
        confidence=answer.confidence,
        details={
            "persona": persona,
            "evidence_chunks_used": len(evidence.results),
            "citations_used": len(answer.citations),
            "repair_iteration": repair_iteration,
        },
    )
    _apply_policy_warning(run, policy)
    return _with_agent_run(
        {"answer_draft": answer, "repair_context": None},
        run,
        persistence,
    )
