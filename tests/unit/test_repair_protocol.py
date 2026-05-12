from models.agent_runs import AgentRun
from models.qa import AnswerDraft, CriticReview
from services.repair import (
    MAX_REPAIR_ITERATIONS,
    build_repair_context,
    is_repair_exhausted,
    latest_agent_run_id,
)


def _answer(repair_iteration: int = 0) -> AnswerDraft:
    return AnswerDraft(
        question="What does the method improve?",
        answer_text="It improves retrieval quality.",
        persona="engineer",
        repair_iteration=repair_iteration,
    )


def test_build_repair_context_propagates_review_and_target():
    review = CriticReview(
        reviewed_answer_id="answer-1",
        unsupported_claims=["Latency claim is unsupported."],
        needs_repair=True,
        repair_target_agent="answer_agent",
        repair_instructions=["Remove the latency claim."],
    )

    context = build_repair_context(review, _answer(), "run-1")

    assert context.original_run_id == "run-1"
    assert context.target_agent == "answer_agent"
    assert context.instructions == ["Remove the latency claim."]
    assert context.iteration == 1
    assert context.critic_review_id == review.id


def test_build_repair_context_increments_answer_iteration():
    review = CriticReview(
        reviewed_answer_id="answer-1",
        needs_repair=True,
        repair_target_agent="answer_agent",
        repair_instructions=["Tighten citations."],
    )

    context = build_repair_context(review, _answer(repair_iteration=1), "run-1")

    assert context.iteration == 2


def test_is_repair_exhausted_uses_max_iterations():
    assert is_repair_exhausted(_answer(repair_iteration=MAX_REPAIR_ITERATIONS - 1)) is False
    assert is_repair_exhausted(_answer(repair_iteration=MAX_REPAIR_ITERATIONS)) is True


def test_latest_agent_run_id_returns_latest_matching_run():
    first = AgentRun(agent_name="answer_agent")
    critic = AgentRun(agent_name="citation_critic")
    second = AgentRun(agent_name="answer_agent")

    assert latest_agent_run_id([first, critic, second], agent_name="answer_agent") == second.id
    assert latest_agent_run_id([first], agent_name="missing") is None
