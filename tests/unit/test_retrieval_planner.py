import json
from unittest.mock import patch

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.retrieval_planner import retrieval_planner_agent
from models.agent_policies import AgentRuntimePolicy
from models.qa import IntentResolution
from models.retrieval import (
    ChunkLocation,
    ChunkSearchResult,
    ChunkSource,
    EvidenceBundle,
    PaperChunk,
    UpsertChunksResult,
)


class RecordingRetrievalLayer:
    def __init__(self, result_sets=None) -> None:
        self.result_sets = list(result_sets or [])
        self.search_queries = []
        self.assemble_calls = []

    def upsert_chunks(self, chunks):
        return UpsertChunksResult(inserted=len(chunks), updated=0, skipped=0)

    def search_chunks(self, query):
        self.search_queries.append(query)
        if self.result_sets:
            return self.result_sets.pop(0)
        return []

    def assemble_evidence(self, query, results, *, max_chunks=5):
        self.assemble_calls.append(
            {"query": query, "results": list(results), "max_chunks": max_chunks}
        )
        selected = list(results)[:max_chunks]
        return EvidenceBundle(
            query=query,
            results=selected,
            coverage_notes=[] if selected else ["no_matching_chunks"],
        )


def _chunk(
    *,
    chunk_id: str = "2310.06825:chunk:0",
    text: str = "The method optimizes a retrieval loss.",
    chunk_type: str = "text",
) -> PaperChunk:
    return PaperChunk(
        id=chunk_id,
        paper_id="2310.06825",
        chunk_index=0,
        text=text,
        chunk_type=chunk_type,
        source=ChunkSource(
            paper_id="2310.06825",
            session_id="session-1",
            arxiv_id="2310.06825",
            title="Example Paper",
        ),
        location=ChunkLocation(page_start=2, page_end=2, section_title="Method"),
    )


def _result(chunk=None) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk=chunk or _chunk(),
        score=0.9,
        rank=1,
        match_reason="test",
    )


def _state(**overrides) -> dict:
    state = {
        "session_id": "session-1",
        "user_message": "What is the loss?",
        "persona": "researcher",
        "intent_resolution": IntentResolution(
            intent="qa_math",
            referenced_paper_ids=["2310.06825"],
            confidence=0.9,
        ),
        "referenced_paper_ids": ["2310.06825"],
    }
    state.update(overrides)
    return state


def _config(layer, persistence=None) -> dict:
    return {
        "configurable": {
            "session_id": "session-1",
            "job_id": "job-1",
            "retrieval_layer": layer,
            "agent_run_persistence": persistence or InMemoryAgentRunPersistence(),
        }
    }


def _replan_payload(**overrides) -> str:
    payload = {
        "search_query": "loss function objective optimization training",
        "chunk_types_priority": ["equation", "text"],
        "section_queries": ["Method", "Training"],
        "k": 12,
        "replanning_reason": "Broadened from exact loss wording.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _run(result: dict):
    runs = result["agent_runs"]
    assert len(runs) == 1
    return runs[0]


def test_retrieval_planner_builds_deterministic_math_plan():
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])

    result = retrieval_planner_agent(_state(), config=_config(layer))

    run = _run(result)
    plan = result["evidence_plan"]
    evidence = result["evidence_bundle"]
    assert plan.intent == "qa_math"
    assert plan.paper_ids == ["2310.06825"]
    assert plan.chunk_types_priority == ["equation", "text"]
    assert plan.section_queries == ["Method", "Objective", "Loss", "Training"]
    assert "What is the loss?" in plan.search_query
    assert "Objective" in plan.search_query
    assert evidence.results[0].chunk.id == "2310.06825:chunk:0"
    assert run.agent_name == "retrieval_planner"
    assert run.model == "claude-sonnet-4-6"
    assert run.status == "completed"
    assert run.output_ref == "state:evidence_bundle"
    assert run.llm_call_count == 0


def test_retrieval_planner_passes_query_filters_to_retrieval_layer():
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])

    retrieval_planner_agent(_state(), config=_config(layer))

    query = layer.search_queries[0]
    assert query.session_id == "session-1"
    assert query.paper_ids == ["2310.06825"]
    assert query.limit == 8
    assert query.filters == {
        "chunk_types_priority": ["equation", "text"],
        "section_queries": ["Method", "Objective", "Loss", "Training"],
    }
    assert layer.assemble_calls[0]["max_chunks"] == 8


def test_retrieval_planner_uses_persona_priorities_for_techlead_comparison():
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])
    state = _state(
        persona="techlead",
        user_message="Compare these papers.",
        intent_resolution=IntentResolution(
            intent="qa_comparison",
            referenced_paper_ids=["2310.06825", "2401.12345"],
        ),
        referenced_paper_ids=["2310.06825", "2401.12345"],
    )

    result = retrieval_planner_agent(state, config=_config(layer))

    plan = result["evidence_plan"]
    assert plan.chunk_types_priority == ["table", "text"]
    assert plan.section_queries == [
        "Results",
        "Experiments",
        "Limitations",
        "Discussion",
    ]
    assert plan.paper_ids == ["2310.06825", "2401.12345"]


def test_retrieval_planner_uses_engineer_priorities_for_factual_qa():
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])
    state = _state(
        persona="engineer",
        user_message="What should I implement?",
        intent_resolution=IntentResolution(
            intent="qa_factual",
            referenced_paper_ids=["2310.06825"],
        ),
        referenced_paper_ids=["2310.06825"],
    )

    result = retrieval_planner_agent(state, config=_config(layer))

    plan = result["evidence_plan"]
    assert plan.chunk_types_priority == ["text", "table"]
    assert plan.section_queries == ["Method", "Implementation", "Results"]


def test_retrieval_planner_uses_researcher_priorities_for_factual_qa():
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])
    state = _state(
        persona="researcher",
        user_message="What is the methodological contribution?",
        intent_resolution=IntentResolution(
            intent="qa_factual",
            referenced_paper_ids=["2310.06825"],
        ),
        referenced_paper_ids=["2310.06825"],
    )

    result = retrieval_planner_agent(state, config=_config(layer))

    plan = result["evidence_plan"]
    assert plan.chunk_types_priority == ["text", "equation"]
    assert plan.section_queries == [
        "Method",
        "Related Work",
        "Limitations",
        "Discussion",
    ]


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_replans_once_when_initial_evidence_empty(mock_call_llm):
    layer = RecordingRetrievalLayer(result_sets=[[], [_result()]])
    mock_call_llm.return_value = (_replan_payload(), None)

    result = retrieval_planner_agent(_state(), config=_config(layer))

    run = _run(result)
    plan = result["evidence_plan"]
    evidence = result["evidence_bundle"]
    assert run.llm_call_count == 1
    assert len(layer.search_queries) == 2
    assert plan.requires_replanning is True
    assert plan.fallback_used is True
    assert plan.k == 12
    assert plan.replanning_reason == "Broadened from exact loss wording."
    assert evidence.results
    assert run.details["iterations_used"] == 2


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_returns_best_effort_empty_evidence_on_replan_error(
    mock_call_llm,
):
    layer = RecordingRetrievalLayer(result_sets=[[]])
    mock_call_llm.return_value = (None, "planner unavailable")

    result = retrieval_planner_agent(_state(), config=_config(layer))

    run = _run(result)
    plan = result["evidence_plan"]
    evidence = result["evidence_bundle"]
    assert evidence.results == []
    assert evidence.coverage_notes == ["no_matching_chunks"]
    assert plan.fallback_used is True
    assert plan.requires_replanning is True
    assert plan.replanning_reason == "planner unavailable"
    assert run.status == "completed"
    assert run.details["fallback_used"] is True


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_marks_fallback_when_replan_parse_fails(mock_call_llm):
    layer = RecordingRetrievalLayer(result_sets=[[]])
    mock_call_llm.return_value = ("not json", None)

    result = retrieval_planner_agent(_state(), config=_config(layer))

    plan = result["evidence_plan"]
    assert plan.fallback_used is True
    assert plan.requires_replanning is True
    assert "JSON parse error" in (plan.replanning_reason or "")


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_marks_fallback_when_replan_shape_invalid(mock_call_llm):
    layer = RecordingRetrievalLayer(result_sets=[[]])
    mock_call_llm.return_value = (
        _replan_payload(chunk_types_priority="text"),
        None,
    )

    result = retrieval_planner_agent(_state(), config=_config(layer))

    plan = result["evidence_plan"]
    assert plan.fallback_used is True
    assert plan.requires_replanning is True
    assert plan.replanning_reason == "chunk_types_priority must be a list"


def test_retrieval_planner_fails_for_non_qa_intent():
    layer = RecordingRetrievalLayer()
    state = _state(
        intent_resolution=IntentResolution(intent="discover", referenced_paper_ids=[]),
        referenced_paper_ids=[],
    )

    result = retrieval_planner_agent(state, config=_config(layer))

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "intent"
    assert result["errors"][0].agent == "retrieval_planner"


def test_retrieval_planner_fails_without_retrieval_layer():
    result = retrieval_planner_agent(
        _state(),
        config={"configurable": {"session_id": "session-1"}},
    )

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "retrieval_layer"


def test_retrieval_planner_fails_without_referenced_papers():
    layer = RecordingRetrievalLayer()
    state = _state(
        intent_resolution=IntentResolution(intent="qa_factual", referenced_paper_ids=[]),
        referenced_paper_ids=[],
    )

    result = retrieval_planner_agent(state, config=_config(layer))

    run = _run(result)
    assert run.status == "failed"
    assert run.details["stage"] == "input"
    assert "referenced_paper_ids" in run.details["error"]


def test_retrieval_planner_records_agent_run_on_success():
    persistence = InMemoryAgentRunPersistence()
    layer = RecordingRetrievalLayer(result_sets=[[_result()]])

    result = retrieval_planner_agent(_state(), config=_config(layer, persistence))

    run = _run(result)
    assert run.input_refs == [
        "state:user_message",
        "state:intent_resolution",
        "state:referenced_paper_ids",
    ]
    assert run.details["policy_applied"]["fallback_strategy"] == (
        "return_best_effort_evidence"
    )
    assert persistence.list_runs() == [run]


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_policy_override_reaches_replan_llm(mock_call_llm):
    layer = RecordingRetrievalLayer(result_sets=[[], [_result()]])
    mock_call_llm.return_value = (_replan_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=4,
        max_tokens=700,
        timeout_seconds=15,
        fallback_strategy="custom_best_effort",
    )
    config = _config(layer)
    config["configurable"]["agent_policy_overrides"] = {
        "retrieval_planner": override
    }

    result = retrieval_planner_agent(_state(), config=config)

    run = _run(result)
    assert run.details["policy_applied"]["fallback_strategy"] == "custom_best_effort"
    assert mock_call_llm.call_args.kwargs["max_tokens"] == 700


@patch("agents.retrieval_planner._call_llm")
def test_retrieval_planner_policy_warning_records_limit_and_actual_calls(
    mock_call_llm,
):
    layer = RecordingRetrievalLayer(result_sets=[[], [_result()]])
    mock_call_llm.return_value = (_replan_payload(), None)
    override = AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=0,
        max_tokens=700,
        timeout_seconds=15,
        fallback_strategy="custom_best_effort",
    )
    config = _config(layer)
    config["configurable"]["agent_policy_overrides"] = {
        "retrieval_planner": override
    }

    result = retrieval_planner_agent(_state(), config=config)

    run = _run(result)
    assert run.status == "completed"
    assert run.details["policy_warning"] == "exceeded_max_tool_calls"
    assert run.details["policy_max_tool_calls"] == 0
    assert run.details["actual_llm_call_count"] == 1
