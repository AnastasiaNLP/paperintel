from __future__ import annotations

import pytest

from evaluation.run_baseline_workspaces import load_baseline_papers


def test_load_baseline_papers_respects_limit():
    papers = load_baseline_papers(
        "golden_dataset/paperintel_30_v0_1.jsonl",
        limit=2,
    )

    assert [paper.paper_id for paper in papers] == ["1810.04805", "2005.14165"]
    assert papers[0].source_url == "https://arxiv.org/abs/1810.04805"


def test_load_baseline_papers_filters_requested_ids_in_request_order():
    papers = load_baseline_papers(
        "golden_dataset/paperintel_30_v0_1.jsonl",
        selected_paper_ids=["1706.03762", "2005.11401"],
    )

    assert [paper.paper_id for paper in papers] == ["1706.03762", "2005.11401"]


def test_load_baseline_papers_rejects_unknown_requested_id():
    with pytest.raises(ValueError, match="not in the golden dataset"):
        load_baseline_papers(
            "golden_dataset/paperintel_30_v0_1.jsonl",
            selected_paper_ids=["missing"],
        )
