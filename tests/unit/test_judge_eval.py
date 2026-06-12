import json
import subprocess
import sys

import pytest

from evaluation.judge_automation import (
    JudgeAutomationError,
    compare_judge_results,
    load_judge_results_jsonl,
    write_judge_results_jsonl,
)
from evaluation.ca_judge_payloads import (
    build_comparison_judge_payload,
    build_synthesis_judge_payload,
)
from evaluation.golden_dataset import load_golden_records
from evaluation.judge_rubrics import EXPECTED_RUBRIC_IDS, load_judge_rubrics
from evaluation.judge_models import JudgeResult, JudgeTask
from evaluation.judge_provider import ConfiguredLLMJudgeProvider, JudgePayload
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


class FailingJudgeProvider:
    def score(self, *, task, rubric, payload):
        raise RuntimeError("provider down")


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
    assert report.status_counts == {"not_scored": 6}
    assert report.average_score is None
    assert report.judge_model is None
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
    assert all(result.task.rubric_version for result in report.results)


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
        judge_model="judge-test",
        dataset_version="seed_5",
        pipeline_version="pipeline-test",
    )

    assert report.mode == "live"
    assert report.judge_model == "judge-test"
    assert report.dataset_version == "seed_5"
    assert report.pipeline_version == "pipeline-test"
    assert report.rubric_versions["recommended_action"].startswith("sha256:")
    assert report.total_tasks == 6
    assert report.scored_tasks == 6
    assert report.status_counts == {"scored": 6}
    assert report.average_score == 0.75
    assert set(report.average_scores_by_rubric) == {
        "recommended_action",
        "implementation_difficulty",
        "action_reasoning",
    }
    assert {result.status for result in report.results} == {"scored"}
    assert {result.score for result in report.results} == {0.75}
    assert all(result.task.sample_id.startswith("report:") for result in report.results)
    assert {result.task.judge_model for result in report.results} == {"judge-test"}
    assert {result.task.dataset_version for result in report.results} == {"seed_5"}
    assert {result.task.pipeline_version for result in report.results} == {
        "pipeline-test"
    }


def test_judge_report_continues_when_provider_raises():
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:1]
    workspaces = load_workspace_records(WORKSPACES_PATH)[:1]
    rubrics = load_judge_rubrics()

    report = build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=FailingJudgeProvider(),
        mode="live",
    )

    assert report.total_tasks == 3
    assert report.scored_tasks == 0
    assert report.status_counts == {"error": 3}
    assert {result.error_code for result in report.results} == {"judge_provider_failed"}


def test_configured_llm_judge_provider_classifies_returned_provider_error(monkeypatch):
    def fake_call_text_llm(**kwargs):
        return None, "judge call failed"

    monkeypatch.setattr("agents.llm_provider.call_text_llm", fake_call_text_llm)
    rubrics = load_judge_rubrics()
    rubric = rubrics["recommended_action"]
    task = _judge_task(rubric)

    result = ConfiguredLLMJudgeProvider().score(
        task=task,
        rubric=rubric,
        payload=_judge_payload(task, rubric),
    )

    assert result.status == "error"
    assert result.error_code == "judge_provider_failed"
    assert result.rationale == "judge call failed"


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
    assert all(item["task"]["sample_id"].startswith("report:") for item in payload["results"])


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


def test_run_judge_eval_writes_jsonl_and_summary_outputs(tmp_path):
    output = tmp_path / "judge_results.jsonl"
    summary = tmp_path / "judge_summary.json"

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
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--dataset-version",
            "seed_5",
            "--pipeline-version",
            "test-pipeline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    results = load_judge_results_jsonl(output)
    assert len(results) == 6
    assert {item.task.dataset_version for item in results} == {"seed_5"}
    assert {item.task.pipeline_version for item in results} == {"test-pipeline"}
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["total_tasks"] == 6
    assert summary_payload["status_counts"] == {"not_scored": 6}


def test_judge_result_jsonl_roundtrip_and_baseline_comparison(tmp_path):
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:1]
    workspaces = load_workspace_records(WORKSPACES_PATH)[:1]
    rubrics = load_judge_rubrics()
    current = build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=FakeScoredJudgeProvider(),
        mode="live",
    ).results
    baseline = []
    for result in current:
        clone = result.model_copy(deep=True)
        clone.score = 0.5
        baseline.append(clone)
    output = tmp_path / "current.jsonl"
    baseline_output = tmp_path / "baseline.jsonl"

    write_judge_results_jsonl(current, output)
    write_judge_results_jsonl(baseline, baseline_output)
    comparison = compare_judge_results(
        current=load_judge_results_jsonl(output),
        baseline=load_judge_results_jsonl(baseline_output),
        min_delta=0.01,
    )

    assert comparison.current_count == 3
    assert comparison.baseline_count == 3
    assert comparison.matched_scored_tasks == 3
    assert len(comparison.improved) == 3
    assert comparison.regressed == []


def test_baseline_comparison_keys_by_task_family_sample_and_rubric():
    rubrics = load_judge_rubrics()
    rubric = rubrics["recommended_action"]
    report_task = _judge_task(rubric, sample_id="shared", task_family="report")
    qa_task = _judge_task(rubric, sample_id="shared", task_family="qa")
    current = [
        JudgeResult(task=report_task, status="scored", score=0.8),
        JudgeResult(task=qa_task, status="scored", score=0.4),
    ]
    baseline = [
        JudgeResult(task=report_task, status="scored", score=0.7),
        JudgeResult(task=qa_task, status="scored", score=0.5),
    ]

    comparison = compare_judge_results(current=current, baseline=baseline)

    assert [(item.task_family, item.sample_id) for item in comparison.improved] == [
        ("report", "shared")
    ]
    assert [(item.task_family, item.sample_id) for item in comparison.regressed] == [
        ("qa", "shared")
    ]


def test_baseline_comparison_rejects_duplicate_scored_keys():
    rubrics = load_judge_rubrics()
    task = _judge_task(rubrics["recommended_action"])
    duplicate = [
        JudgeResult(task=task, status="scored", score=0.8),
        JudgeResult(task=task, status="scored", score=0.7),
    ]

    with pytest.raises(JudgeAutomationError, match="Duplicate scored judge result"):
        compare_judge_results(current=duplicate, baseline=[])


def test_run_judge_eval_writes_baseline_comparison(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    comparison_output = tmp_path / "comparison.json"
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:1]
    workspaces = load_workspace_records(WORKSPACES_PATH)[:1]
    rubrics = load_judge_rubrics()
    baseline_results = build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=FakeScoredJudgeProvider(),
        mode="live",
    ).results
    for result in baseline_results:
        result.score = 0.5
    write_judge_results_jsonl(baseline_results, baseline)

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
            "--baseline",
            str(baseline),
            "--compare-output",
            str(comparison_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(comparison_output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert payload["current_count"] == 6
    assert payload["baseline_count"] == 3
    assert payload["matched_scored_tasks"] == 0
    assert len(payload["missing_in_current"]) == 3


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


def _judge_task(
    rubric,
    *,
    sample_id: str = "report:paper-0",
    task_family: str = "report",
) -> JudgeTask:
    return JudgeTask(
        rubric_id=rubric.rubric_id,
        paper_id="paper-0",
        sample_id=sample_id,
        task_family=task_family,
        input_refs=["paper_workspace:paper-0:finalized_report_json"],
        rubric_hash=rubric.sha256,
        rubric_version=f"sha256:{rubric.sha256[:12]}",
        mode="live",
        judge_model="judge-test",
    )


def _judge_payload(task: JudgeTask, rubric) -> JudgePayload:
    return JudgePayload(
        paper_id=task.paper_id,
        title="Paper 0",
        rubric_id=rubric.rubric_id,
        rubric_text=rubric.text,
        finalized_report_json={"recommended_action": "prototype"},
    )
