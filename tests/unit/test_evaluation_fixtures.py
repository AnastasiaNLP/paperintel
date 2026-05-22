from evaluation.deterministic_metrics import evaluate_workspace
from evaluation.fixtures import build_partial_workspace, build_perfect_workspace
from evaluation.golden_dataset import load_golden_records


def test_build_perfect_workspace_passes_deterministic_checks():
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]
    workspace = build_perfect_workspace(record)

    result = evaluate_workspace(record, workspace)

    assert workspace.finalized_report_json is not None
    assert workspace.finalized_report_json["recommended_action"] == "prototype"
    assert workspace.finalized_report_json["implementation_difficulty"] in {
        "trivial",
        "moderate",
        "significant",
        "research_only",
    }
    assert "action_reasoning" in workspace.finalized_report_json
    assert result.passed
    assert result.score == 1.0


def test_build_partial_workspace_keeps_valid_workspace_but_fails_some_checks():
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0]

    workspace = build_partial_workspace(record)
    result = evaluate_workspace(record, workspace)

    assert workspace.paper_id == record.paper_id
    assert not result.passed
    assert 0 < result.score < 1
