from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact
from models.synthesis import (
    SynthesisAgentResult,
    SynthesisCitation,
    SynthesisRecommendation,
    SynthesisReport,
)
from evaluation.ca_structural_checks import (
    check_comparison_artifact,
    check_synthesis_result,
)


def _comparison_artifact(*, producer: str = "comparison_analyst") -> ComparisonArtifact:
    return ComparisonArtifact(
        session_id="session-1",
        paper_ids=["paper-0", "paper-1"],
        comparison_report_json={
            "producer": producer,
            "papers_summary": [
                {
                    "paper_index": 0,
                    "input_url": "https://arxiv.org/abs/paper-0",
                    "completed": True,
                    "title": "Paper Zero",
                    "arxiv_id": "paper-0",
                },
                {
                    "paper_index": 1,
                    "input_url": "https://arxiv.org/abs/paper-1",
                    "completed": True,
                    "title": "Paper One",
                    "arxiv_id": "paper-1",
                },
            ],
            "comparison_matrix": [
                {
                    "task": "MMLU",
                    "metric": "Accuracy",
                    "values_by_paper": {0: 72.0, 1: None},
                    "units_by_paper": {0: "%", 1: "%"},
                    "conditions_by_paper": {0: "zero-shot", 1: None},
                }
            ],
            "trade_offs": "Paper 0 has reported benchmark evidence.",
            "recommendations": [],
            "overall_winner_index": None,
            "overall_winner_reasoning": "No clear winner.",
            "winner_basis": "no_clear_winner",
        },
        comparison_markdown="# Comparison",
    )


def _synthesis_result(*, citations: list[SynthesisCitation] | None = None) -> SynthesisAgentResult:
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
            summary="Engineering synthesis.",
            key_takeaways=["Paper 0 is more mature."],
            trade_offs=["Benchmark evidence differs."],
            recommended_next_steps=[
                SynthesisRecommendation(
                    recommendation="Prototype Paper 0.",
                    reasoning="It has stronger artifact evidence.",
                )
            ],
            citations=citations
            if citations is not None
            else [
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


def test_valid_comparison_artifact_passes_structural_checks():
    result = check_comparison_artifact(
        _comparison_artifact(),
        requested_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is True
    assert result.errors == []


def test_comparison_fails_if_producer_is_wrong():
    result = check_comparison_artifact(
        _comparison_artifact(producer="batch_comparator"),
        requested_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("producer" in error for error in result.errors)


def test_comparison_fails_if_requested_paper_missing_from_artifact():
    artifact = _comparison_artifact()
    artifact.paper_ids = ["paper-0"]

    result = check_comparison_artifact(
        artifact,
        requested_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("artifact.paper_ids" in error for error in result.errors)


def test_comparison_fails_if_matrix_drops_requested_paper():
    artifact = _comparison_artifact()
    artifact.comparison_report_json["comparison_matrix"][0]["values_by_paper"] = {
        0: 72.0
    }

    result = check_comparison_artifact(
        artifact,
        requested_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("comparison_matrix" in error for error in result.errors)


def test_valid_synthesis_result_passes_structural_checks():
    result = check_synthesis_result(
        _synthesis_result(),
        selected_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is True
    assert result.errors == []


def test_synthesis_fails_on_unknown_citation_paper_id():
    result = check_synthesis_result(
        _synthesis_result(
            citations=[
                SynthesisCitation(
                    paper_id="paper-0",
                    quote_or_summary="Paper 0 summary.",
                ),
                SynthesisCitation(
                    paper_id="paper-x",
                    quote_or_summary="Unknown paper.",
                ),
            ]
        ),
        selected_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("unknown paper ids" in error for error in result.errors)


def test_synthesis_fails_when_selected_paper_has_no_citation():
    result = check_synthesis_result(
        _synthesis_result(
            citations=[
                SynthesisCitation(
                    paper_id="paper-0",
                    quote_or_summary="Paper 0 summary.",
                )
            ]
        ),
        selected_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("missing citations" in error for error in result.errors)


def test_synthesis_fails_when_policy_applied_is_missing():
    synthesis = _synthesis_result()
    synthesis.agent_run.details.clear()

    result = check_synthesis_result(
        synthesis,
        selected_paper_ids=["paper-0", "paper-1"],
    )

    assert result.passed is False
    assert any("policy_applied" in error for error in result.errors)
