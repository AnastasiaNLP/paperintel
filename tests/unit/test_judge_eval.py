import json
import subprocess
import sys

import pytest

from evaluation.ca_judge_payloads import (
    build_comparison_judge_payload,
    build_synthesis_judge_payload,
)
from evaluation.golden_dataset import load_golden_records
from evaluation.judge_rubrics import EXPECTED_RUBRIC_IDS, load_judge_rubrics
from evaluation.judge_models import JudgeResult
from evaluation.judge_runner import build_dry_run_judge_report, build_judge_report
from evaluation.runner import load_workspace_records
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.synthesis import (
    SynthesisAgentResult,
    SynthesisCitation,
    SynthesisRecommendation,
    SynthesisReport,
)


WORKSPACES_PATH = "tests/fixtures/evaluation/workspaces_seed_sample.jsonl"


class FakeScoredJudgeProvider:
    def score(self, *, task, rubric, payload):
        return JudgeResult(
            task=task,
            status="scored",
            score=0.75,
            rationale=f"Fake score for {payload.paper_id} with {rubric.rubric_id}.",
        )


def test_load_judge_rubrics_finds_expected_versioned_files():
    rubrics = load_judge_rubrics()

    assert set(rubrics) == EXPECTED_RUBRIC_IDS
    assert all(len(rubric.sha256) == 64 for rubric in rubrics.values())
    assert rubrics["recommended_action"].text.startswith("# Rubric:")
    assert rubrics["comparison_balance"].text.startswith("# Rubric:")
    assert rubrics["synthesis_persona_fit"].text.startswith("# Rubric:")


def test_dry_run_judge_report_builds_report_tasks_for_matched_workspaces():
    records = load_golden_records("golden_dataset/seed_5.jsonl")
    workspaces = load_workspace_records(WORKSPACES_PATH)
    rubrics = load_judge_rubrics()

    report = build_dry_run_judge_report(records, workspaces, rubrics)

    assert report.mode == "dry_run"
    assert report.total_tasks == 6
    assert report.scored_tasks == 0
    assert {result.status for result in report.results} == {"not_scored"}
    assert {result.task.paper_id for result in report.results} == {
        "1706.03762",
        "2005.11401",
    }
    assert {result.task.rubric_id for result in report.results} == {
        "recommended_action",
        "implementation_difficulty",
        "action_reasoning",
    }
    assert all(result.task.rubric_hash for result in report.results)


def test_live_mode_runner_uses_provider_without_gate_semantics():
    records = load_golden_records("golden_dataset/seed_5.jsonl")
    workspaces = load_workspace_records(WORKSPACES_PATH)
    rubrics = load_judge_rubrics()

    report = build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=FakeScoredJudgeProvider(),
        mode="live",
    )

    assert report.mode == "live"
    assert report.total_tasks == 6
    assert report.scored_tasks == 6
    assert {result.status for result in report.results} == {"scored"}
    assert {result.score for result in report.results} == {0.75}


def test_report_rubrics_skip_when_finalized_report_is_missing():
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]
    workspace = PaperWorkspace(
        session_id="session-1",
        paper_id=record.paper_id,
        title=record.title,
        source_url=record.source_url,
        pipeline_stage="chunk_and_index",
        finalized_report_json=None,
    )
    rubrics = load_judge_rubrics()

    report = build_judge_report(
        records=[record],
        workspaces=[workspace],
        rubrics=rubrics,
        provider=FakeScoredJudgeProvider(),
        mode="live",
    )

    assert report.total_tasks == 3
    assert report.scored_tasks == 0
    assert {result.status for result in report.results} == {"skipped"}
    assert all("No finalized_report_json" in result.rationale for result in report.results)


def test_run_judge_eval_dry_run_cli_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_judge_eval",
            "--golden",
            "golden_dataset/seed_5.jsonl",
            "--workspaces",
            WORKSPACES_PATH,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["mode"] == "dry_run"
    assert payload["total_tasks"] == 6
    assert payload["scored_tasks"] == 0
    assert {item["status"] for item in payload["results"]} == {"not_scored"}
    assert all(item["task"]["rubric_hash"] for item in payload["results"])


def test_run_judge_eval_requires_exactly_one_mode_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_judge_eval",
            "--golden",
            "golden_dataset/seed_5.jsonl",
            "--workspaces",
            WORKSPACES_PATH,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "choose exactly one" in result.stdout


def test_ca_comparison_judge_payload_includes_only_selected_workspaces():
    artifact = ComparisonArtifact(
        id="cmp-1",
        session_id="session-1",
        paper_ids=["paper-1", "paper-0"],
        comparison_report_json={"producer": "comparison_analyst"},
        comparison_markdown="# Comparison",
    )
    payload = build_comparison_judge_payload(
        artifact=artifact,
        workspaces=[
            _workspace("paper-0"),
            _workspace("paper-1"),
            _workspace("paper-x"),
        ],
    )

    assert payload["selected_paper_ids"] == ["paper-1", "paper-0"]
    assert [workspace["paper_id"] for workspace in payload["workspaces"]] == [
        "paper-1",
        "paper-0",
    ]
    assert "paper-x" not in {
        workspace["paper_id"] for workspace in payload["workspaces"]
    }
    assert payload["comparison_artifact"]["comparison_markdown"] == "# Comparison"


def test_ca_comparison_judge_payload_raises_if_selected_workspace_missing():
    artifact = ComparisonArtifact(
        id="cmp-1",
        session_id="session-1",
        paper_ids=["paper-0", "paper-1"],
        comparison_report_json={"producer": "comparison_analyst"},
        comparison_markdown="# Comparison",
    )

    with pytest.raises(ValueError, match="paper-1"):
        build_comparison_judge_payload(
            artifact=artifact,
            workspaces=[_workspace("paper-0")],
        )


def test_ca_synthesis_judge_payload_includes_persona_and_optional_comparison():
    comparison = ComparisonArtifact(
        id="cmp-1",
        session_id="session-1",
        paper_ids=["paper-0", "paper-1"],
        comparison_report_json={"producer": "comparison_analyst"},
        comparison_markdown="# Comparison",
    )
    payload = build_synthesis_judge_payload(
        result=_synthesis_result(),
        workspaces=[
            _workspace("paper-0"),
            _workspace("paper-1"),
            _workspace("paper-x"),
        ],
        comparison=comparison,
    )

    assert payload["persona"] == "engineer"
    assert payload["selected_paper_ids"] == ["paper-0", "paper-1"]
    assert [workspace["paper_id"] for workspace in payload["workspaces"]] == [
        "paper-0",
        "paper-1",
    ]
    assert payload["comparison_context"]["id"] == "cmp-1"
    assert "paper-x" not in {
        workspace["paper_id"] for workspace in payload["workspaces"]
    }


def test_ca_synthesis_judge_payload_raises_if_selected_workspace_missing():
    with pytest.raises(ValueError, match="paper-1"):
        build_synthesis_judge_payload(
            result=_synthesis_result(),
            workspaces=[_workspace("paper-0")],
        )


def test_ca_synthesis_judge_payload_omits_unrelated_comparison_context():
    payload = build_synthesis_judge_payload(
        result=_synthesis_result(),
        workspaces=[_workspace("paper-0"), _workspace("paper-1")],
        comparison=ComparisonArtifact(
            id="cmp-unrelated",
            session_id="session-1",
            paper_ids=["paper-x", "paper-y"],
            comparison_markdown="# Unrelated",
        ),
    )

    assert "comparison_context" not in payload


def _workspace(paper_id: str) -> PaperWorkspace:
    return PaperWorkspace(
        session_id="session-1",
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        source_url=f"https://arxiv.org/abs/{paper_id}",
        pipeline_stage="completed",
        finalized_report_json={
            "executive_summary": f"{paper_id} summary",
            "key_innovation": "innovation",
            "practical_implications": "implications",
            "implementation_difficulty": "moderate",
            "recommended_action": "prototype",
            "action_reasoning": "reasoning",
        },
        method_extraction_json={
            "method_name": f"Method {paper_id}",
            "description": "description",
            "novelty_claim": "novelty",
            "key_components": ["component"],
            "compared_to": ["baseline"],
            "limitations_stated": ["limitation"],
        },
        benchmarks_json=[
            {
                "task": "MMLU",
                "metric": "Accuracy",
                "value": 72.0,
            }
        ],
        readiness_json={
            "has_open_code": True,
            "framework_integrations": ["PyTorch"],
            "dependencies": ["torch"],
            "maturity_level": "experimental",
            "maturity_reasoning": "reasoning",
        },
        full_markdown_report="# Report",
    )


def _synthesis_result() -> SynthesisAgentResult:
    run = AgentRun(
        agent_name="synthesis_agent",
        session_id="session-1",
        input_refs=["paper_workspace:paper-0", "paper_workspace:paper-1"],
    )
    run.complete(
        output_ref="synthesis_report",
        details={"policy_applied": {"max_tokens": 4000}},
    )
    return SynthesisAgentResult(
        report=SynthesisReport(
            persona="engineer",
            summary="Synthesis summary.",
            key_takeaways=["Takeaway."],
            trade_offs=["Trade-off."],
            recommended_next_steps=[
                SynthesisRecommendation(
                    recommendation="Prototype.",
                    reasoning="Evidence supports a small trial.",
                )
            ],
            citations=[
                SynthesisCitation(
                    paper_id="paper-0",
                    quote_or_summary="Paper 0 summary.",
                ),
                SynthesisCitation(
                    paper_id="paper-1",
                    quote_or_summary="Paper 1 summary.",
                ),
            ],
        ),
        response_text="Synthesis for engineer",
        agent_run=run,
    )
