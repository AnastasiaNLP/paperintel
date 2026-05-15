import graph_conversation
from graph_conversation import build_conversation_graph
from models.agent_runs import AgentRun
from models.qa import (
    AnswerDraft,
    CriticReview,
    EvidencePlan,
    IntentResolution,
    RepairContext,
)
from models.retrieval import (
    ChunkLocation,
    ChunkSearchResult,
    ChunkSource,
    CitationRef,
    EvidenceBundle,
    PaperChunk,
)


def _run(agent_name: str) -> AgentRun:
    return AgentRun(agent_name=agent_name, session_id="session-1").complete(
        output_ref=f"state:{agent_name}",
    )


def _chunk() -> PaperChunk:
    return PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="The paper introduces a retrieval method for grounded QA.",
        chunk_type="text",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id="session-1",
            arxiv_id="2310.06825",
            title="Grounded QA Paper",
        ),
        location=ChunkLocation(page_start=2, page_end=2, section_title="Method"),
    )


def _citation() -> CitationRef:
    return CitationRef(
        paper_id="2310.06825",
        chunk_id="2310.06825:chunk:0",
        page_start=2,
        page_end=2,
        section_title="Method",
    )


def _evidence_bundle() -> EvidenceBundle:
    chunk = _chunk()
    return EvidenceBundle(
        query="grounded QA method",
        results=[ChunkSearchResult(chunk=chunk, score=0.9, rank=1)],
        citations=[_citation()],
    )


def _answer(*, repair_iteration: int = 0) -> AnswerDraft:
    return AnswerDraft(
        id=f"answer-{repair_iteration}",
        question="What is the method?",
        answer_text="The paper introduces a retrieval method for grounded QA.",
        citations=[_citation()],
        persona="engineer",
        confidence=0.8,
        repair_iteration=repair_iteration,
    )


def _initial_state() -> dict:
    return {
        "session_id": "session-1",
        "user_message": "What is the method?",
        "persona": "engineer",
        "agent_runs": [],
        "errors": [],
    }


def _patch_happy_path(monkeypatch, calls: list[str]) -> None:
    def fake_router(state, config=None):
        calls.append("intent_router")
        return {
            "intent": "qa_factual",
            "referenced_paper_ids": ["2310.06825"],
            "intent_resolution": IntentResolution(
                intent="qa_factual",
                referenced_paper_ids=["2310.06825"],
                confidence=0.95,
            ),
            "persona": state["persona"],
            "agent_runs": [_run("intent_router")],
        }

    def fake_planner(state, config=None):
        calls.append("retrieval_planner")
        return {
            "evidence_plan": EvidencePlan(
                intent=state["intent"],
                paper_ids=state["referenced_paper_ids"],
                search_query="grounded QA method",
                chunk_types_priority=["text"],
                k=4,
            ),
            "evidence_bundle": _evidence_bundle(),
            "agent_runs": [_run("retrieval_planner")],
        }

    def fake_answer(state, config=None):
        calls.append("answer_agent")
        assert state["evidence_bundle"].results
        return {
            "answer_draft": _answer(),
            "repair_context": None,
            "agent_runs": [_run("answer_agent")],
        }

    def fake_critic(state, config=None):
        calls.append("citation_critic")
        assert state["answer_draft"].citations
        assert state["evidence_bundle"].results
        return {
            "critic_review": CriticReview(
                reviewed_answer_id=state["answer_draft"].id,
                needs_repair=False,
                critic_confidence=0.9,
            ),
            "repair_context": None,
            "agent_runs": [_run("citation_critic")],
        }

    monkeypatch.setattr(graph_conversation, "intent_router_agent", fake_router)
    monkeypatch.setattr(graph_conversation, "retrieval_planner_agent", fake_planner)
    monkeypatch.setattr(graph_conversation, "answer_agent", fake_answer)
    monkeypatch.setattr(graph_conversation, "citation_critic_agent", fake_critic)


def _agent_names(result: dict) -> list[str]:
    return [run.agent_name for run in result["agent_runs"]]


def test_conversation_graph_happy_path_runs_four_agents(monkeypatch):
    calls: list[str] = []
    _patch_happy_path(monkeypatch, calls)

    result = build_conversation_graph().invoke(_initial_state())

    assert calls == [
        "intent_router",
        "retrieval_planner",
        "answer_agent",
        "citation_critic",
    ]
    assert result["answer_draft"].answer_text
    assert result["critic_review"].needs_repair is False
    assert result["repair_context"] is None
    assert result["evidence_bundle"].results
    assert _agent_names(result) == calls


def test_conversation_graph_clarification_path_stops_before_planner(monkeypatch):
    calls: list[str] = []

    def fake_router(state, config=None):
        calls.append("intent_router")
        return {
            "intent": "clarification_needed",
            "needs_clarification": True,
            "clarification_question": "Which paper do you mean?",
            "intent_resolution": IntentResolution(
                intent="clarification_needed",
                ambiguous=True,
                clarification_question="Which paper do you mean?",
            ),
            "agent_runs": [_run("intent_router")],
        }

    monkeypatch.setattr(graph_conversation, "intent_router_agent", fake_router)

    result = build_conversation_graph().invoke(_initial_state())

    assert calls == ["intent_router"]
    assert result["clarification_question"] == "Which paper do you mean?"
    assert result["needs_clarification"] is True
    assert "evidence_plan" not in result
    assert "answer_draft" not in result
    assert "critic_review" not in result
    assert _agent_names(result) == ["intent_router"]


def test_conversation_graph_analyze_paper_path_sets_needs_analysis(monkeypatch):
    calls: list[str] = []

    def fake_router(state, config=None):
        calls.append("intent_router")
        return {
            "intent": "analyze_paper",
            "intent_resolution": IntentResolution(
                intent="analyze_paper",
                confidence=0.9,
            ),
            "agent_runs": [_run("intent_router")],
        }

    monkeypatch.setattr(graph_conversation, "intent_router_agent", fake_router)

    result = build_conversation_graph().invoke(_initial_state())

    assert calls == ["intent_router"]
    assert result["needs_analysis"] is True
    assert "send the paper URL directly" in result["clarification_question"]
    assert "evidence_plan" not in result
    assert "answer_draft" not in result
    assert "critic_review" not in result
    assert _agent_names(result) == ["intent_router"]


def test_conversation_graph_repair_loop_reruns_answer_once(monkeypatch):
    calls: list[str] = []
    answer_calls = 0
    critic_calls = 0

    def fake_router(state, config=None):
        calls.append("intent_router")
        return {
            "intent": "qa_factual",
            "referenced_paper_ids": ["2310.06825"],
            "intent_resolution": IntentResolution(
                intent="qa_factual",
                referenced_paper_ids=["2310.06825"],
            ),
            "agent_runs": [_run("intent_router")],
        }

    def fake_planner(state, config=None):
        calls.append("retrieval_planner")
        return {
            "evidence_plan": EvidencePlan(
                intent="qa_factual",
                paper_ids=["2310.06825"],
                search_query="grounded QA method",
            ),
            "evidence_bundle": _evidence_bundle(),
            "agent_runs": [_run("retrieval_planner")],
        }

    def fake_answer(state, config=None):
        nonlocal answer_calls
        answer_calls += 1
        calls.append("answer_agent")
        repair_context = state.get("repair_context")
        if repair_context is None:
            answer = _answer(repair_iteration=0)
        else:
            assert repair_context.iteration == 1
            answer = _answer(repair_iteration=repair_context.iteration)
        return {
            "answer_draft": answer,
            "repair_context": None,
            "agent_runs": [_run("answer_agent")],
        }

    def fake_critic(state, config=None):
        nonlocal critic_calls
        critic_calls += 1
        calls.append("citation_critic")
        answer = state["answer_draft"]
        if critic_calls == 1:
            review = CriticReview(
                reviewed_answer_id=answer.id,
                unsupported_claims=["Unsupported deployment claim."],
                needs_repair=True,
                repair_target_agent="answer_agent",
                repair_instructions=["Remove unsupported deployment claim."],
            )
            return {
                "critic_review": review,
                "repair_context": RepairContext(
                    original_run_id="answer-run-1",
                    target_agent="answer_agent",
                    instructions=review.repair_instructions,
                    iteration=1,
                    critic_review_id=review.id,
                ),
                "answer_draft": None,
                "agent_runs": [_run("citation_critic")],
            }

        assert answer.repair_iteration == 1
        return {
            "critic_review": CriticReview(
                reviewed_answer_id=answer.id,
                needs_repair=False,
            ),
            "repair_context": None,
            "agent_runs": [_run("citation_critic")],
        }

    monkeypatch.setattr(graph_conversation, "intent_router_agent", fake_router)
    monkeypatch.setattr(graph_conversation, "retrieval_planner_agent", fake_planner)
    monkeypatch.setattr(graph_conversation, "answer_agent", fake_answer)
    monkeypatch.setattr(graph_conversation, "citation_critic_agent", fake_critic)

    result = build_conversation_graph().invoke(_initial_state())

    assert answer_calls == 2
    assert critic_calls == 2
    assert result["repair_context"] is None
    assert result["answer_draft"].repair_iteration == 1
    assert _agent_names(result) == [
        "intent_router",
        "retrieval_planner",
        "answer_agent",
        "citation_critic",
        "answer_agent",
        "citation_critic",
    ]
    assert calls == _agent_names(result)


def test_conversation_graph_repair_context_none_terminates_after_acceptance(
    monkeypatch,
):
    calls: list[str] = []
    answer_calls = 0
    critic_calls = 0

    def fake_router(state, config=None):
        return {
            "intent": "qa_factual",
            "referenced_paper_ids": ["2310.06825"],
            "intent_resolution": IntentResolution(
                intent="qa_factual",
                referenced_paper_ids=["2310.06825"],
            ),
            "agent_runs": [_run("intent_router")],
        }

    def fake_planner(state, config=None):
        return {
            "evidence_plan": EvidencePlan(
                intent="qa_factual",
                paper_ids=["2310.06825"],
                search_query="grounded QA method",
            ),
            "evidence_bundle": _evidence_bundle(),
            "agent_runs": [_run("retrieval_planner")],
        }

    def fake_answer(state, config=None):
        nonlocal answer_calls
        answer_calls += 1
        calls.append("answer_agent")
        iteration = 0
        if state.get("repair_context") is not None:
            iteration = state["repair_context"].iteration
        return {
            "answer_draft": _answer(repair_iteration=iteration),
            "repair_context": None,
            "agent_runs": [_run("answer_agent")],
        }

    def fake_critic(state, config=None):
        nonlocal critic_calls
        critic_calls += 1
        calls.append("citation_critic")
        if critic_calls == 1:
            review = CriticReview(
                reviewed_answer_id=state["answer_draft"].id,
                unsupported_claims=["Unsupported claim."],
                needs_repair=True,
                repair_target_agent="answer_agent",
                repair_instructions=["Remove unsupported claim."],
            )
            return {
                "critic_review": review,
                "repair_context": RepairContext(
                    original_run_id="answer-run-1",
                    target_agent="answer_agent",
                    instructions=["Remove unsupported claim."],
                    iteration=1,
                    critic_review_id=review.id,
                ),
                "answer_draft": None,
                "agent_runs": [_run("citation_critic")],
            }
        return {
            "critic_review": CriticReview(
                reviewed_answer_id=state["answer_draft"].id,
                needs_repair=False,
            ),
            "repair_context": None,
            "agent_runs": [_run("citation_critic")],
        }

    monkeypatch.setattr(graph_conversation, "intent_router_agent", fake_router)
    monkeypatch.setattr(graph_conversation, "retrieval_planner_agent", fake_planner)
    monkeypatch.setattr(graph_conversation, "answer_agent", fake_answer)
    monkeypatch.setattr(graph_conversation, "citation_critic_agent", fake_critic)

    result = build_conversation_graph().invoke(_initial_state())

    assert answer_calls == 2
    assert critic_calls == 2
    assert result["repair_context"] is None
    assert calls == [
        "answer_agent",
        "citation_critic",
        "answer_agent",
        "citation_critic",
    ]
