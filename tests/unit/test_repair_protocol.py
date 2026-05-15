from models.agent_runs import AgentRun
from models.qa import AnswerDraft, CriticReview, RepairContext
from services.repair import (
    GENERIC_REPAIR_INSTRUCTION,
    MAX_REPAIR_ITERATIONS,
    build_repair_input_refs,
    build_repair_context,
    is_repair_exhausted,
    latest_agent_run_id,
    normalize_repair_decision,
    should_trigger_repair,
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


def test_normalize_preserves_issues_overriding_false_flag():
    needs_repair, instructions = normalize_repair_decision(
        needs_repair=False,
        unsupported_claims=["Unsupported claim."],
        missing_evidence=[],
        contradictions=[],
        repair_instructions=["Remove unsupported claim."],
    )

    assert needs_repair is True
    assert instructions == ["Remove unsupported claim."]


def test_normalize_ignores_noisy_flag_without_issues_or_instructions():
    needs_repair, instructions = normalize_repair_decision(
        needs_repair=True,
        unsupported_claims=[],
        missing_evidence=[],
        contradictions=[],
        repair_instructions=[],
    )

    assert needs_repair is False
    assert instructions == []


def test_normalize_keeps_flag_when_explicit_instructions_present():
    needs_repair, instructions = normalize_repair_decision(
        needs_repair=True,
        unsupported_claims=[],
        missing_evidence=[],
        contradictions=[],
        repair_instructions=["Tighten the answer scope."],
    )

    assert needs_repair is True
    assert instructions == ["Tighten the answer scope."]


def test_normalize_adds_generic_when_issues_but_no_instructions():
    needs_repair, instructions = normalize_repair_decision(
        needs_repair=True,
        unsupported_claims=[],
        missing_evidence=["Missing evidence for latency."],
        contradictions=[],
        repair_instructions=[],
    )

    assert needs_repair is True
    assert instructions == [GENERIC_REPAIR_INSTRUCTION]


def test_normalize_preserves_explicit_instructions_with_issues():
    needs_repair, instructions = normalize_repair_decision(
        needs_repair=True,
        unsupported_claims=[],
        missing_evidence=[],
        contradictions=["Contradicts chunk 3."],
        repair_instructions=["Remove the contradiction."],
    )

    assert needs_repair is True
    assert instructions == ["Remove the contradiction."]


def test_should_trigger_repair_returns_true_with_issues_and_flag():
    review = CriticReview(
        reviewed_answer_id="answer-1",
        unsupported_claims=["Unsupported claim."],
        needs_repair=True,
        repair_target_agent="answer_agent",
        repair_instructions=["Remove unsupported claim."],
    )

    assert should_trigger_repair(review) is True


def test_should_trigger_repair_returns_false_without_issues():
    review = CriticReview(reviewed_answer_id="answer-1")

    assert should_trigger_repair(review) is False


def test_should_trigger_repair_returns_false_when_flag_false():
    review = CriticReview(
        reviewed_answer_id="answer-1",
        unsupported_claims=["Informational only."],
        needs_repair=False,
    )

    assert should_trigger_repair(review) is False


def test_build_repair_input_refs_includes_critic_review_id():
    context = RepairContext(
        original_run_id="run-1",
        target_agent="answer_agent",
        instructions=["Repair answer."],
        iteration=1,
        critic_review_id="review-1",
    )

    refs = build_repair_input_refs(["state:user_message"], context)

    assert refs == [
        "state:user_message",
        "critic_review:review-1",
        "repair_iteration:1",
    ]


def test_build_repair_input_refs_preserves_original_refs_order():
    context = RepairContext(
        original_run_id="run-1",
        target_agent="answer_agent",
        instructions=["Repair answer."],
        iteration=2,
        critic_review_id="review-2",
    )

    refs = build_repair_input_refs(["state:user_message", "state:evidence_bundle"], context)

    assert refs[:2] == ["state:user_message", "state:evidence_bundle"]
    assert refs[-2:] == ["critic_review:review-2", "repair_iteration:2"]
