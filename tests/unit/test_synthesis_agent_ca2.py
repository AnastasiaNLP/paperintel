import json
from unittest.mock import patch

import pytest

from agents.agent_run_recorder import InMemoryAgentRunPersistence
from agents.synthesis_agent import synthesize_workspaces
from models.agent_policies import AgentRuntimePolicy
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.session import Session
from models.synthesis import SynthesisAgentResult
from services.paperintel_service import (
    NotEnoughPapersForComparisonError,
    PaperIntelService,
    PaperWorkspaceNotFoundError,
    PaperWorkspaceNotReadyError,
)


def _workspace(
    paper_id: str,
    *,
    title: str,
    method_name: str,
    stage: str = "completed",
) -> PaperWorkspace:
    return PaperWorkspace(
        session_id="session-1",
        paper_id=paper_id,
        title=title,
        source_url=f"https://arxiv.org/abs/{paper_id}",
        pipeline_stage=stage,
        method_extraction_json={
            "method_name": method_name,
            "description": f"{method_name} method",
            "novelty_claim": f"{method_name} novelty",
            "key_components": ["component"],
            "compared_to": ["baseline"],
            "limitations_stated": ["limitation"],
        },
        benchmarks_json=[
            {
                "task": "MMLU",
                "metric": "Accuracy",
                "value": 72.0,
                "unit": "%",
                "baseline_comparison": None,
                "conditions": "zero-shot",
            }
        ],
        readiness_json={
            "has_open_code": True,
            "code_url": f"https://github.com/example/{paper_id}",
            "huggingface_model": None,
            "framework_integrations": ["PyTorch"],
            "min_gpu_requirement": None,
            "estimated_inference_cost": None,
            "dependencies": ["torch"],
            "maturity_level": "experimental",
            "maturity_reasoning": f"{paper_id} readiness",
        },
        finalized_report_json={
            "executive_summary": f"{method_name} summary",
            "key_innovation": f"{method_name} innovation",
            "practical_implications": f"{method_name} implications",
            "implementation_difficulty": "moderate",
            "recommended_action": "prototype",
            "action_reasoning": f"{method_name} action reasoning",
        },
        full_markdown_report=f"# {title}",
    )


def _llm_report(persona: str = "engineer", *, extra_citation: bool = False) -> str:
    citations = [
        {"paper_id": "paper-0", "quote_or_summary": "Method Zero summary"},
        {"paper_id": "paper-1", "quote_or_summary": "Method One summary"},
    ]
    if extra_citation:
        citations.append({"paper_id": "paper-x", "quote_or_summary": "Ignore me"})
    return json.dumps(
        {
            "persona": persona,
            "summary": "Use Paper 0 first and watch Paper 1.",
            "key_takeaways": ["Paper 0 is easier to prototype.", "Paper 1 is riskier."],
            "trade_offs": ["Quality and maturity differ."],
            "recommended_next_steps": [
                {
                    "recommendation": "Prototype Paper 0.",
                    "reasoning": "It has clearer readiness evidence.",
                }
            ],
            "citations": citations,
            "limitations": ["Benchmark extraction is thin."],
        }
    )


class FakeStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def require_session(self, session_id: str) -> Session:
        assert session_id == self.session.id
        return self.session


class FakeHandler:
    def __init__(self, session: Session) -> None:
        self.store = FakeStore(session)
        self.agent_run_persistence = InMemoryAgentRunPersistence()
        self.messages = []

    def handle_message(self, session_id: str, message: str):
        self.messages.append((session_id, message))
        raise AssertionError("synthesize_papers must not call handle_message")


class FakeRepository:
    def __init__(
        self,
        workspaces: list[PaperWorkspace],
        *,
        comparison: ComparisonArtifact | None = None,
    ) -> None:
        self.workspaces = list(workspaces)
        self.comparison = comparison
        self.saved: list[ComparisonArtifact] = []

    def list_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        return [
            workspace
            for workspace in self.workspaces
            if workspace.session_id == session_id
        ]

    def get_workspace(self, session_id: str, paper_id: str) -> PaperWorkspace | None:
        for workspace in self.list_workspaces(session_id):
            if workspace.paper_id == paper_id:
                return workspace
        return None

    def latest_comparison(self, session_id: str) -> ComparisonArtifact | None:
        if self.comparison and self.comparison.session_id == session_id:
            return self.comparison
        return None

    def save_comparison(self, artifact: ComparisonArtifact) -> ComparisonArtifact:
        self.saved.append(artifact)
        return artifact


def _service(
    workspaces: list[PaperWorkspace],
    *,
    active_ids: list[str],
    persona: str = "engineer",
    comparison: ComparisonArtifact | None = None,
) -> PaperIntelService:
    session = Session(
        id="session-1",
        persona=persona,  # type: ignore[arg-type]
        active_paper_ids=active_ids,
    )
    handler = FakeHandler(session)
    return PaperIntelService(
        handler=handler,  # type: ignore[arg-type]
        artifact_repository=FakeRepository(workspaces, comparison=comparison),
    )


@patch("agents.synthesis_agent._call_llm")
def test_ca2_synthesis_agent_builds_report_records_run_and_filters_citations(mock_call_llm):
    mock_call_llm.return_value = (_llm_report(extra_citation=True), None)
    persistence = InMemoryAgentRunPersistence()
    comparison = ComparisonArtifact(
        id="cmp-1",
        session_id="session-1",
        paper_ids=["paper-0", "paper-1"],
        comparison_markdown="# Comparison",
    )

    result = synthesize_workspaces(
        session_id="session-1",
        persona="researcher",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        prompt="Focus on novelty.",
        comparison=comparison,
        config={
            "configurable": {
                "session_id": "session-1",
                "agent_run_persistence": persistence,
            }
        },
    )

    assert result.report.persona == "researcher"
    assert {citation.paper_id for citation in result.report.citations} == {
        "paper-0",
        "paper-1",
    }
    assert "Synthesis for researcher" in result.response_text
    assert result.agent_run.agent_name == "synthesis_agent"
    assert result.agent_run.status == "completed"
    assert result.agent_run.details["policy_applied"]["max_tokens"] == 4000
    assert result.agent_run.input_refs == [
        "paper_workspace:paper-0",
        "paper_workspace:paper-1",
        "comparison_artifact:cmp-1",
    ]
    assert persistence.list_runs() == [result.agent_run]


def test_ca2_synthesis_agent_uses_policy_max_tokens(monkeypatch):
    captured = {}

    def fake_call_text_llm(**kwargs):
        captured.update(kwargs)
        return _llm_report(), None

    monkeypatch.setattr("agents.synthesis_agent.call_text_llm", fake_call_text_llm)

    synthesize_workspaces(
        session_id="session-1",
        persona="engineer",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        config={
            "configurable": {
                "agent_policy_overrides": {
                    "synthesis_agent": AgentRuntimePolicy(
                        max_iterations=1,
                        max_tool_calls=1,
                        max_tokens=1234,
                        timeout_seconds=60,
                        fallback_strategy="durable_artifact_summary_fallback",
                    )
                }
            }
        },
    )

    assert captured["max_tokens"] == 1234


@patch("agents.synthesis_agent._call_llm")
def test_ca2_synthesis_agent_works_without_comparison_and_falls_back(mock_call_llm):
    mock_call_llm.return_value = (None, "provider unavailable")

    result = synthesize_workspaces(
        session_id="session-1",
        persona="techlead",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
    )

    assert result.report.persona == "techlead"
    assert result.report.citations[0].paper_id == "paper-0"
    assert result.agent_run.status == "fallback_used"
    assert result.agent_run.details["fallback_reason"] == "provider unavailable"
    assert result.agent_run.input_refs == [
        "paper_workspace:paper-0",
        "paper_workspace:paper-1",
    ]


@patch("agents.synthesis_agent._call_llm")
def test_ca2_synthesis_agent_timeout_fallback_uses_timeout_reason(mock_call_llm):
    mock_call_llm.return_value = (None, "Synthesis Agent LLM call timed out")

    result = synthesize_workspaces(
        session_id="session-1",
        persona="techlead",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
    )

    assert result.agent_run.status == "fallback_used"
    assert result.agent_run.termination_reason == "timeout"
    assert (
        result.agent_run.details["fallback_reason"]
        == "Synthesis Agent LLM call timed out"
    )


@patch("agents.synthesis_agent._call_llm")
def test_ca2_service_synthesize_uses_active_ids_and_not_qa_wrapper(mock_call_llm):
    mock_call_llm.return_value = (_llm_report(), None)
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        active_ids=["paper-0", "paper-1"],
    )

    result = service.synthesize_papers("session-1", prompt="Compare risks.")

    assert isinstance(result, SynthesisAgentResult)
    assert service.handler.messages == []
    assert result.agent_run.details["paper_ids"] == ["paper-0", "paper-1"]
    assert service.artifact_repository.saved == []


@patch("agents.synthesis_agent._call_llm")
def test_ca2_service_synthesize_supports_explicit_ids_and_dedupes(mock_call_llm):
    mock_call_llm.return_value = (_llm_report(), None)
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        active_ids=["paper-1", "paper-0"],
    )

    result = service.synthesize_papers(
        "session-1",
        paper_ids=["paper-0", "paper-0", "paper-1"],
    )

    assert result.agent_run.details["paper_ids"] == ["paper-0", "paper-1"]


def test_ca2_service_synthesize_rejects_fewer_than_two_papers():
    service = _service(
        [_workspace("paper-0", title="Paper Zero", method_name="Method Zero")],
        active_ids=["paper-0"],
    )

    with pytest.raises(NotEnoughPapersForComparisonError):
        service.synthesize_papers("session-1")


def test_ca2_service_synthesize_rejects_missing_workspace():
    service = _service(
        [_workspace("paper-0", title="Paper Zero", method_name="Method Zero")],
        active_ids=["paper-0", "missing"],
    )

    with pytest.raises(PaperWorkspaceNotFoundError):
        service.synthesize_papers("session-1")


def test_ca2_service_synthesize_rejects_failed_workspace():
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace(
                "paper-1",
                title="Paper One",
                method_name="Method One",
                stage="failed",
            ),
        ],
        active_ids=["paper-0", "paper-1"],
    )

    with pytest.raises(PaperWorkspaceNotReadyError):
        service.synthesize_papers("session-1")


@patch("agents.synthesis_agent._call_llm")
def test_ca2_service_synthesize_uses_latest_comparison_only_if_relevant(mock_call_llm):
    mock_call_llm.return_value = (_llm_report(), None)
    relevant = ComparisonArtifact(
        id="cmp-relevant",
        session_id="session-1",
        paper_ids=["paper-1", "paper-x"],
        comparison_markdown="# Relevant",
    )
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        active_ids=["paper-0", "paper-1"],
        comparison=relevant,
    )

    result = service.synthesize_papers("session-1")

    assert "comparison_artifact:cmp-relevant" in result.agent_run.input_refs

    unrelated = ComparisonArtifact(
        id="cmp-unrelated",
        session_id="session-1",
        paper_ids=["paper-x", "paper-y"],
        comparison_markdown="# Unrelated",
    )
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace("paper-1", title="Paper One", method_name="Method One"),
        ],
        active_ids=["paper-0", "paper-1"],
        comparison=unrelated,
    )

    result = service.synthesize_papers("session-1")

    assert "comparison_artifact:cmp-unrelated" not in result.agent_run.input_refs


@patch("agents.comparison_analyst._call_llm")
def test_ca2_service_compare_rejects_failed_workspace(mock_call_llm):
    service = _service(
        [
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero"),
            _workspace(
                "paper-1",
                title="Paper One",
                method_name="Method One",
                stage="paper_failure_finalize",
            ),
        ],
        active_ids=["paper-0", "paper-1"],
    )

    with pytest.raises(PaperWorkspaceNotReadyError):
        service.compare_papers("session-1")
