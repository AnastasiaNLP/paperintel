import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.comparator import comparator_agent
from agents.comparison_analyst import compare_workspaces
from agents.agent_run_recorder import InMemoryAgentRunPersistence
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.agent_policies import AgentRuntimePolicy
from models.schemas import ComparisonReport
from models.session import Session
from services.paperintel_service import (
    NotEnoughPapersForComparisonError,
    PaperIntelService,
    PaperWorkspaceNotFoundError,
)


def _workspace(
    paper_id: str,
    *,
    title: str,
    method_name: str,
    benchmark_value: float,
    maturity: str = "experimental",
) -> PaperWorkspace:
    return PaperWorkspace(
        session_id="session-1",
        paper_id=paper_id,
        title=title,
        source_url=f"https://arxiv.org/abs/{paper_id}",
        pipeline_stage="chunk_and_index",
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
                "value": benchmark_value,
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
            "maturity_level": maturity,
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


def _claims(winner: int = 0) -> str:
    return json.dumps(
        {
            "trade_offs": "Paper 0 is stronger on the shared benchmark; Paper 1 is close.",
            "recommendations": [
                {
                    "constraint": "best benchmark evidence",
                    "recommended_paper_index": winner,
                    "reasoning": "It has the stronger MMLU accuracy in the durable workspace.",
                }
            ],
            "overall_winner_index": winner,
            "overall_winner_reasoning": "The winner has stronger benchmark evidence.",
            "winner_basis": "benchmark_dominant",
        }
    )


class FakeStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def require_session(self, session_id: str) -> Session:
        assert session_id == self.session.id
        return self.session


class FakeRepository:
    def __init__(self, workspaces: list[PaperWorkspace]) -> None:
        self.workspaces = list(workspaces)
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
        return self.saved[-1] if self.saved else None

    def save_comparison(self, artifact: ComparisonArtifact) -> ComparisonArtifact:
        self.saved.append(artifact)
        return artifact


def _service(workspaces: list[PaperWorkspace], *, active_ids: list[str]) -> PaperIntelService:
    session = Session(id="session-1", active_paper_ids=active_ids)
    handler = SimpleNamespace(
        store=FakeStore(session),
        agent_run_persistence=InMemoryAgentRunPersistence(),
    )
    return PaperIntelService(
        handler=handler,  # type: ignore[arg-type]
        artifact_repository=FakeRepository(workspaces),
    )


@patch("agents.comparator._call_llm")
def test_ca1_batch_comparator_sets_batch_producer(mock_call_llm):
    mock_call_llm.return_value = (_claims(), None)

    result = comparator_agent(
        {
            "papers": [
                {
                    "paper_index": 0,
                    "input_url": "https://arxiv.org/abs/a",
                    "benchmarks": [
                        {
                            "task": "MMLU",
                            "metric": "Accuracy",
                            "value": 70.0,
                        }
                    ],
                    "completed": True,
                },
                {
                    "paper_index": 1,
                    "input_url": "https://arxiv.org/abs/b",
                    "benchmarks": [
                        {
                            "task": "MMLU",
                            "metric": "Accuracy",
                            "value": 65.0,
                        }
                    ],
                    "completed": True,
                },
            ]
        }
    )

    assert result["comparison_report"].producer == "batch_comparator"


@patch("agents.comparison_analyst._call_llm")
def test_ca1_comparison_analyst_builds_report_and_agent_run(mock_call_llm):
    mock_call_llm.return_value = (_claims(winner=0), None)
    persistence = InMemoryAgentRunPersistence()

    result = compare_workspaces(
        session_id="session-1",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
            _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=70.0),
        ],
        prompt="Prefer benchmark strength.",
        config={
            "configurable": {
                "session_id": "session-1",
                "agent_run_persistence": persistence,
            }
        },
    )

    assert result.report.producer == "comparison_analyst"
    assert result.report.overall_winner_index == 0
    assert "Paper Comparison" in result.markdown
    assert result.agent_run.agent_name == "comparison_analyst"
    assert result.agent_run.status == "completed"
    assert result.agent_run.details["policy_applied"]["max_tokens"] == 4000
    assert result.agent_run.input_refs == [
        "paper_workspace:paper-0",
        "paper_workspace:paper-1",
    ]
    assert persistence.list_runs() == [result.agent_run]


def test_ca1_comparison_analyst_uses_policy_max_tokens(monkeypatch):
    captured = {}

    def fake_call_text_llm(**kwargs):
        captured.update(kwargs)
        return _claims(), None

    monkeypatch.setattr(
        "agents.comparison_analyst.call_text_llm",
        fake_call_text_llm,
    )

    compare_workspaces(
        session_id="session-1",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
            _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=70.0),
        ],
        config={
            "configurable": {
                "agent_policy_overrides": {
                    "comparison_analyst": AgentRuntimePolicy(
                        max_iterations=1,
                        max_tool_calls=1,
                        max_tokens=1234,
                        timeout_seconds=60,
                        fallback_strategy="deterministic_comparison_fallback",
                    )
                }
            }
        },
    )

    assert captured["max_tokens"] == 1234


@patch("agents.comparison_analyst._call_llm")
def test_ca1_comparison_analyst_falls_back_when_llm_unavailable(mock_call_llm):
    mock_call_llm.return_value = (None, "provider unavailable")

    result = compare_workspaces(
        session_id="session-1",
        workspaces=[
            _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
            _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=70.0),
        ],
    )

    assert isinstance(result.report, ComparisonReport)
    assert result.report.producer == "comparison_analyst"
    assert result.report.trade_offs
    assert result.agent_run.status == "fallback_used"
    assert result.agent_run.details["fallback_reason"] == "provider unavailable"


@patch("agents.comparison_analyst._call_llm")
def test_ca1_service_compare_papers_loads_saves_and_preserves_order(mock_call_llm):
    mock_call_llm.return_value = (_claims(winner=1), None)
    workspaces = [
        _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
        _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=80.0),
    ]
    service = _service(workspaces, active_ids=["paper-0", "paper-1"])

    artifact = service.compare_papers(
        "session-1",
        paper_ids=["paper-1", "paper-0"],
        prompt="Compare in requested order.",
    )

    assert artifact.paper_ids == ["paper-1", "paper-0"]
    assert artifact.comparison_report_json["producer"] == "comparison_analyst"
    assert "Paper Comparison" in artifact.comparison_markdown
    assert service.artifact_repository.saved == [artifact]
    assert len(service.handler.agent_run_persistence.list_runs()) == 1


@patch("agents.comparison_analyst._call_llm")
def test_ca1_service_compare_papers_uses_active_ids_by_default(mock_call_llm):
    mock_call_llm.return_value = (_claims(), None)
    workspaces = [
        _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
        _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=70.0),
    ]
    service = _service(workspaces, active_ids=["paper-0", "paper-1"])

    artifact = service.compare_papers("session-1")

    assert artifact.paper_ids == ["paper-0", "paper-1"]
    assert service.artifact_repository.saved == [artifact]


@patch("agents.comparison_analyst._call_llm")
def test_ca1_service_compare_papers_deduplicates_requested_ids(mock_call_llm):
    mock_call_llm.return_value = (_claims(), None)
    workspaces = [
        _workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0),
        _workspace("paper-1", title="Paper One", method_name="Method One", benchmark_value=70.0),
    ]
    service = _service(workspaces, active_ids=["paper-0", "paper-1"])

    artifact = service.compare_papers(
        "session-1",
        paper_ids=["paper-0", "paper-0", "paper-1"],
    )

    assert artifact.paper_ids == ["paper-0", "paper-1"]


def test_ca1_service_compare_papers_requires_two_papers():
    service = _service(
        [_workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0)],
        active_ids=["paper-0"],
    )

    with pytest.raises(NotEnoughPapersForComparisonError):
        service.compare_papers("session-1")


def test_ca1_service_compare_papers_requires_existing_workspaces():
    service = _service(
        [_workspace("paper-0", title="Paper Zero", method_name="Method Zero", benchmark_value=75.0)],
        active_ids=["paper-0", "missing"],
    )

    with pytest.raises(PaperWorkspaceNotFoundError):
        service.compare_papers("session-1")
