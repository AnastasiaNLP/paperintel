import os

import pytest

from evaluation.golden_dataset import load_golden_records
from evaluation.judge_provider import ConfiguredLLMJudgeProvider
from evaluation.judge_rubrics import load_judge_rubrics
from evaluation.judge_runner import build_judge_report
from evaluation.runner import load_workspace_records


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is required for live judge eval",
)
def test_live_judge_eval_scores_seed_fixture():
    records = load_golden_records("golden_dataset/seed_5.jsonl")[:1]
    workspaces = load_workspace_records(
        "tests/fixtures/evaluation/workspaces_seed_sample.jsonl"
    )[:1]
    rubrics = load_judge_rubrics()
    provider = ConfiguredLLMJudgeProvider(max_tokens=500)

    report = build_judge_report(
        records=records,
        workspaces=workspaces,
        rubrics=rubrics,
        provider=provider,
        mode="live",
    )

    assert report.mode == "live"
    assert report.total_tasks == 3
    assert report.scored_tasks == 3
    assert {result.status for result in report.results} == {"scored"}
    for result in report.results:
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0
        assert result.rationale
