from models.agent_runs import AgentRun
from models.qa import AnswerDraft, CriticReview, RepairContext


MAX_REPAIR_ITERATIONS = 2
GENERIC_REPAIR_INSTRUCTION = (
    "Rewrite the answer so every substantive claim is supported by the provided chunks."
)


def normalize_repair_decision(
    *,
    needs_repair: bool,
    unsupported_claims: list[str],
    missing_evidence: list[str],
    contradictions: list[str],
    repair_instructions: list[str],
) -> tuple[bool, list[str]]:
    """
    Normalize a critic's repair signal into a deterministic repair decision.

    LLM critics can emit noisy flags. Concrete issue lists always trigger repair,
    even if the raw flag is false. A bare needs_repair=true without issues or
    explicit instructions is ignored because it is not actionable.
    """
    has_review_issues = bool(unsupported_claims or missing_evidence or contradictions)

    if has_review_issues and not needs_repair:
        needs_repair = True

    if needs_repair and not has_review_issues and not repair_instructions:
        return False, []

    if needs_repair and has_review_issues and not repair_instructions:
        return True, [GENERIC_REPAIR_INSTRUCTION]

    return needs_repair, repair_instructions


def should_trigger_repair(review: CriticReview) -> bool:
    """Return whether a normalized critic review should start a repair pass."""
    if not review.needs_repair:
        return False
    return bool(
        review.repair_instructions
        or review.unsupported_claims
        or review.missing_evidence
        or review.contradictions
    )


def build_repair_context(
    review: CriticReview,
    answer_draft: AnswerDraft,
    original_run_id: str,
) -> RepairContext:
    target_agent = review.repair_target_agent or "answer_agent"
    return RepairContext(
        original_run_id=original_run_id,
        target_agent=target_agent,
        instructions=review.repair_instructions,
        iteration=answer_draft.repair_iteration + 1,
        critic_review_id=review.id,
    )


def build_repair_input_refs(
    original_refs: list[str],
    repair_context: RepairContext,
) -> list[str]:
    return [
        *original_refs,
        f"critic_review:{repair_context.critic_review_id}",
        f"repair_iteration:{repair_context.iteration}",
    ]


def is_repair_exhausted(answer_draft: AnswerDraft) -> bool:
    return answer_draft.repair_iteration >= MAX_REPAIR_ITERATIONS


def latest_agent_run_id(
    runs: list[AgentRun],
    *,
    agent_name: str,
) -> str | None:
    for run in reversed(runs):
        if run.agent_name == agent_name:
            return run.id
    return None
