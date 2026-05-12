from models.agent_runs import AgentRun
from models.qa import AnswerDraft, CriticReview, RepairContext


MAX_REPAIR_ITERATIONS = 2


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
