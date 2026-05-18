import pytest

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore, SessionNotFoundError
from models.discovery import SearchCandidate
from models.errors import ErrorCodes, StructuredError
from models.qa import AnswerDraft
from models.schemas import (
    BenchmarkResult,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)
from services.selection_parser import SelectionHandler


class FakeRunner:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {
            "response_text": "assistant response",
            "intent": "qa",
            "referenced_paper_ids": ["paper-1"],
            "artifact_refs": ["artifact-1"],
            "next_phase": "qa",
        }
        self.error = error
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        if self.error is not None:
            raise self.error
        return self.result


class FakeRetrievalLayer:
    pass


class FakeCandidateRepository:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.updated = []

    def list_latest_for_session(self, session_id: str):
        return list(self.candidates)

    def update_status(self, candidate_id: str, status: str):
        self.updated.append((candidate_id, status))
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate.model_copy(update={"status": status})
        return None


class FakeArtifactRepository:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.workspaces = []
        self.comparisons = []

    def upsert_workspace(self, workspace):
        if self.error is not None:
            raise self.error
        self.workspaces.append(workspace)
        return workspace

    def save_comparison(self, artifact):
        if self.error is not None:
            raise self.error
        self.comparisons.append(artifact)
        return artifact


def _candidate(rank: int) -> SearchCandidate:
    arxiv_id = f"2401.0000{rank}"
    return SearchCandidate(
        id=f"candidate-{rank}",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=rank,
        title=f"Paper {rank}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
        year=2024,
    )


def _paper_slot(
    *,
    paper_index: int = 0,
    arxiv_id: str = "2401.00001",
    input_url: str | None = None,
    completed: bool = True,
) -> PaperSlot:
    return PaperSlot(
        paper_index=paper_index,
        input_url=input_url or f"https://arxiv.org/abs/{arxiv_id}",
        metadata=PaperMetadata(
            title=f"Paper {arxiv_id}",
            authors=["A. Author"],
            arxiv_id=arxiv_id,
            published_date="2024-01-01",
            abstract="Abstract.",
            categories=["cs.CL"],
        ),
        method_extraction=MethodExtraction(
            method_name="Method",
            description="Description.",
            novelty_claim="Novelty.",
            key_components=["component"],
            compared_to=["baseline"],
            limitations_stated=["limitation"],
        ),
        benchmarks=[
            BenchmarkResult(
                task="translation",
                metric="BLEU",
                value=42.0,
            )
        ],
        production_readiness=ProductionReadiness(
            has_open_code=True,
            code_url="https://github.com/example/repo",
            huggingface_model=None,
            framework_integrations=["PyTorch"],
            min_gpu_requirement="A100",
            estimated_inference_cost=None,
            dependencies=["torch"],
            maturity_level="experimental",
            maturity_reasoning="Prototype quality.",
        ),
        engineer_report=EngineerReport(
            executive_summary="Summary.",
            key_innovation="Innovation.",
            practical_implications="Implications.",
            implementation_difficulty="moderate",
            recommended_action="prototype",
            action_reasoning="Worth prototyping.",
        ),
        markdown_report="# Report",
        completed=completed,
    )


def _handler(
    runner=None,
    persistence=None,
    analysis_runner=None,
    retrieval_layer=None,
    artifact_repository=None,
):
    store = InMemorySessionStore()
    runner = runner or FakeRunner()
    persistence = persistence or InMemoryAgentRunPersistence()
    return (
        ChatHandler(
            store=store,
            conversation_runner=runner,
            analysis_runner=analysis_runner,
            agent_run_persistence=persistence,
            retrieval_layer=retrieval_layer,
            artifact_repository=artifact_repository,
        ),
        store,
        runner,
        persistence,
    )


def test_create_session_explicitly_sets_phase_and_persona():
    handler, store, _, _ = _handler()

    session = handler.create_session(
        persona="researcher",
        original_query="agent memory",
    )

    stored = store.require_session(session.id)
    assert stored.id == session.id
    assert stored.phase == "idle"
    assert stored.persona == "researcher"
    assert stored.original_query == "agent memory"


def test_handle_message_requires_existing_session():
    handler, _, _, _ = _handler()

    with pytest.raises(SessionNotFoundError):
        handler.handle_message("missing", "hello")


def test_handle_message_writes_user_turn_before_graph_and_assistant_after():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is in this paper?")

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "What is in this paper?"
    assert turns[1].content == "assistant response"
    assert turns[1].intent == "qa"
    assert turns[1].referenced_paper_ids == ["paper-1"]
    assert turns[1].artifact_refs == ["artifact-1"]
    assert result.user_turn_id == turns[0].id
    assert result.assistant_turn_id == turns[1].id
    assert runner.calls[0]["input"]["user_message"] == "What is in this paper?"
    assert "message" not in runner.calls[0]["input"]


def test_handle_message_updates_phase_from_graph_result():
    handler, store, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "answer this")

    assert result.phase == "qa"
    assert store.require_session(session.id).phase == "qa"


def test_handler_propagates_session_id_and_persistence_to_graph_config():
    persistence = InMemoryAgentRunPersistence()
    retrieval_layer = FakeRetrievalLayer()
    handler, store, runner, _ = _handler(
        persistence=persistence,
        retrieval_layer=retrieval_layer,
    )
    session = handler.create_session()
    store.add_active_paper(session.id, "paper-1")

    handler.handle_message(session.id, "hello")

    call = runner.calls[0]
    assert call["input"]["session_id"] == session.id
    assert call["input"]["persona"] == "engineer"
    assert call["input"]["referenced_paper_ids"] == ["paper-1"]
    assert call["config"]["configurable"]["session_id"] == session.id
    assert call["config"]["configurable"]["session_store"] is store
    assert call["config"]["configurable"]["agent_run_persistence"] is persistence
    assert call["config"]["configurable"]["retrieval_layer"] is retrieval_layer


def test_graph_failure_preserves_user_turn_and_writes_structured_error_turn():
    runner = FakeRunner(error=RuntimeError("boom"))
    handler, store, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "fail please")

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "fail please"
    assert isinstance(turns[1].error, StructuredError)
    assert turns[1].error.code == ErrorCodes.FATAL_ERROR
    assert turns[1].error.session_id == session.id
    assert turns[1].error.details["exception_type"] == "RuntimeError"
    assert result.phase == "failed"
    assert result.error == turns[1].error
    assert result.errors == [turns[1].error]
    assert store.require_session(session.id).phase == "failed"


def test_two_messages_in_same_session_accumulate_turn_history():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    handler.handle_message(session.id, "first")
    handler.handle_message(session.id, "second")

    turns = store.list_recent_turns(session.id)
    assert [turn.content for turn in turns] == [
        "first",
        "assistant response",
        "second",
        "assistant response",
    ]
    assert len(runner.calls) == 2
    assert runner.calls[1]["input"]["user_message"] == "second"


def test_graph_invocation_result_and_handler_result_are_separate_types():
    handler, _, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "hello")

    assert hasattr(result, "user_turn_id")
    assert not hasattr(result, "raw")


def test_conversation_answer_draft_becomes_response_text():
    answer = AnswerDraft(
        question="What is the method?",
        answer_text="It uses retrieval.",
        persona="engineer",
        citations=[],
    )
    runner = FakeRunner(result={"answer_draft": answer, "intent": "qa_factual"})
    handler, store, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "What is the method?")

    assert result.response_text == "It uses retrieval."
    assert result.citations == []
    assert store.list_recent_turns(session.id)[1].content == "It uses retrieval."


def test_conversation_clarification_becomes_response_text():
    runner = FakeRunner(
        result={
            "intent": "clarification_needed",
            "needs_clarification": True,
            "clarification_question": "Which paper do you mean?",
        }
    )
    handler, _, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "What about the second paper?")

    assert result.response_text == "Which paper do you mean?"
    assert result.intent == "clarification_needed"


def test_analyze_paper_without_analysis_runner_returns_controlled_response():
    handler, store, runner, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2310.06825")

    assert runner.calls == []
    assert result.intent == "analyze_paper"
    assert result.needs_analysis is True
    assert "analysis is configured" in result.response_text
    assert [turn.role for turn in store.list_recent_turns(session.id)] == [
        "user",
        "assistant",
    ]


def test_analyze_selected_papers_invokes_analysis_batch():
    analysis_runner = FakeRunner(
        result={
            "full_markdown_report": "# Selected analysis complete",
            "next_phase": "qa",
        }
    )
    handler, store, _, _ = _handler(analysis_runner=analysis_runner)
    session = handler.create_session()

    result = handler.analyze_selected_papers(
        session.id,
        [
            "https://arxiv.org/abs/2401.00001",
            "https://arxiv.org/abs/2401.00002",
        ],
    )

    assert len(analysis_runner.calls) == 1
    state = analysis_runner.calls[0]["input"]
    assert state["input_type"] == "url"
    assert state["input_value"] == "https://arxiv.org/abs/2401.00001"
    assert state["batch_urls"] == [
        "https://arxiv.org/abs/2401.00001",
        "https://arxiv.org/abs/2401.00002",
    ]
    assert state["total_papers"] == 2
    assert state["current_paper_index"] == 0
    assert result.intent == "analyze_paper"
    assert result.phase == "qa"
    assert result.response_text == "# Selected analysis complete"
    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "Analyze selected papers"
    assert turns[0].intent == "analyze_paper"


def test_handler_persists_single_analysis_workspace():
    artifact_repository = FakeArtifactRepository()
    analysis_runner = FakeRunner(
        result={
            "papers": [_paper_slot(arxiv_id="2401.00001")],
            "full_markdown_report": "# Analysis complete",
            "processing_stage": "chunk_and_index",
            "next_phase": "qa",
        }
    )
    handler, _, _, _ = _handler(
        analysis_runner=analysis_runner,
        artifact_repository=artifact_repository,
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2401.00001")

    assert result.phase == "qa"
    assert not result.errors
    assert len(artifact_repository.workspaces) == 1
    workspace = artifact_repository.workspaces[0]
    assert workspace.session_id == session.id
    assert workspace.paper_id == "2401.00001"
    assert workspace.title == "Paper 2401.00001"
    assert workspace.source_url == "https://arxiv.org/abs/2401.00001"
    assert workspace.pipeline_stage == "chunk_and_index"
    assert workspace.method_extraction_json["method_name"] == "Method"
    assert workspace.benchmarks_json == [
        {
            "task": "translation",
            "metric": "BLEU",
            "value": 42.0,
            "unit": None,
            "baseline_comparison": None,
            "conditions": None,
        }
    ]
    assert workspace.readiness_json["maturity_level"] == "experimental"
    assert workspace.finalized_report_json["recommended_action"] == "prototype"
    assert workspace.full_markdown_report == "# Report"


def test_analyze_selected_papers_persists_batch_workspaces_and_comparison():
    artifact_repository = FakeArtifactRepository()
    analysis_runner = FakeRunner(
        result={
            "papers": [
                _paper_slot(paper_index=0, arxiv_id="2401.00001"),
                _paper_slot(paper_index=1, arxiv_id="2401.00002"),
            ],
            "comparison_report": {"winner_basis": "readiness"},
            "comparison_markdown": "# Paper Comparison",
            "processing_stage": "comparison_completed",
            "next_phase": "qa",
        }
    )
    handler, _, _, _ = _handler(
        analysis_runner=analysis_runner,
        artifact_repository=artifact_repository,
    )
    session = handler.create_session()

    result = handler.analyze_selected_papers(
        session.id,
        [
            "https://arxiv.org/abs/2401.00001",
            "https://arxiv.org/abs/2401.00002",
        ],
    )

    assert result.phase == "qa"
    assert [workspace.paper_id for workspace in artifact_repository.workspaces] == [
        "2401.00001",
        "2401.00002",
    ]
    assert len(artifact_repository.comparisons) == 1
    comparison = artifact_repository.comparisons[0]
    assert comparison.session_id == session.id
    assert comparison.paper_ids == ["2401.00001", "2401.00002"]
    assert comparison.comparison_report_json == {"winner_basis": "readiness"}
    assert comparison.comparison_markdown == "# Paper Comparison"


def test_handler_skips_artifact_persistence_for_failed_analysis():
    artifact_repository = FakeArtifactRepository()
    analysis_runner = FakeRunner(
        result={
            "papers": [_paper_slot(arxiv_id="2401.00001")],
            "processing_stage": "failed",
            "paper_failed": True,
            "paper_failure_reason": "metadata failed",
        }
    )
    handler, _, _, _ = _handler(
        analysis_runner=analysis_runner,
        artifact_repository=artifact_repository,
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2401.00001")

    assert result.phase == "failed"
    assert artifact_repository.workspaces == []
    assert artifact_repository.comparisons == []


def test_handler_artifact_persistence_failure_returns_warning():
    artifact_repository = FakeArtifactRepository(error=RuntimeError("db down"))
    analysis_runner = FakeRunner(
        result={
            "papers": [_paper_slot(arxiv_id="2401.00001")],
            "full_markdown_report": "# Analysis complete",
            "processing_stage": "chunk_and_index",
            "next_phase": "qa",
        }
    )
    handler, _, _, _ = _handler(
        analysis_runner=analysis_runner,
        artifact_repository=artifact_repository,
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "https://arxiv.org/abs/2401.00001")

    assert result.phase == "qa"
    assert result.errors
    assert result.errors[0].code == ErrorCodes.WARNING
    assert "Artifact persistence failed" in result.errors[0].message


def test_analyze_selected_papers_without_analysis_runner_returns_controlled_response():
    handler, store, _, _ = _handler()
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.analyze_selected_papers(
        session.id,
        ["https://arxiv.org/abs/2401.00001"],
    )

    assert result.intent == "analyze_paper"
    assert result.needs_analysis is True
    assert result.phase == "selection"
    assert "configure analysis" in result.response_text
    assert [turn.role for turn in store.list_recent_turns(session.id)] == [
        "user",
        "assistant",
    ]


def test_analyze_selected_papers_graph_failure_preserves_turns():
    analysis_runner = FakeRunner(error=RuntimeError("analysis crash"))
    handler, store, _, _ = _handler(analysis_runner=analysis_runner)
    session = handler.create_session()

    result = handler.analyze_selected_papers(
        session.id,
        ["https://arxiv.org/abs/2401.00001"],
    )

    turns = store.list_recent_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "Analyze selected papers"
    assert turns[1].error is not None
    assert result.phase == "failed"
    assert result.intent == "analyze_paper"
    assert result.errors


def test_conversation_discovery_signal_becomes_controlled_response():
    runner = FakeRunner(
        result={
            "intent": "discover",
            "needs_discovery": True,
            "discovery_topic": "long context memory for agents",
        }
    )
    handler, _, _, _ = _handler(runner=runner)
    session = handler.create_session()

    result = handler.handle_message(session.id, "Can you help with this topic?")

    assert result.intent == "discover"
    assert result.needs_discovery is True
    assert result.discovery_topic == "long context memory for agents"
    assert "Discovery is not configured" in result.response_text


def test_handler_routes_selection_phase_to_selection_handler():
    store = InMemorySessionStore()
    runner = FakeRunner()
    repository = FakeCandidateRepository([_candidate(1), _candidate(2), _candidate(3)])
    selection_handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )
    handler = ChatHandler(
        store=store,
        conversation_runner=runner,
        selection_handler=selection_handler,
    )
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 1 and 3")

    assert runner.calls == []
    assert result.intent == "select_papers"
    assert result.phase == "idle"
    assert result.referenced_paper_ids == ["2401.00001", "2401.00003"]
    assert repository.updated == [
        ("candidate-1", "selected"),
        ("candidate-3", "selected"),
    ]
    assert store.require_session(session.id).selected_candidate_ids == [
        "candidate-1",
        "candidate-3",
    ]
    assert "Selected 2 papers" in result.response_text


def test_handler_keeps_selection_phase_on_invalid_selection():
    store = InMemorySessionStore()
    runner = FakeRunner()
    repository = FakeCandidateRepository([_candidate(1), _candidate(2)])
    selection_handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )
    handler = ChatHandler(
        store=store,
        conversation_runner=runner,
        selection_handler=selection_handler,
    )
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 9")

    assert runner.calls == []
    assert result.phase == "selection"
    assert repository.updated == []
    assert "Available numbers" in result.response_text


def test_handler_selection_phase_without_selection_handler_returns_controlled_response():
    handler, store, runner, _ = _handler()
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "use 1")

    assert runner.calls == []
    assert result.intent == "select_papers"
    assert result.phase == "selection"
    assert "not configured" in result.response_text
