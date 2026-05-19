import json

import pytest

from evaluation.export_workspaces import (
    WorkspaceExportError,
    dump_workspaces_jsonl,
    export_workspaces_for_session,
)
from models.artifacts import PaperWorkspace


class FakeWorkspaceRepository:
    def __init__(self, workspaces):
        self.workspaces = workspaces
        self.session_ids = []

    def list_workspaces(self, session_id):
        self.session_ids.append(session_id)
        return list(self.workspaces)


def _workspace(paper_id: str) -> PaperWorkspace:
    return PaperWorkspace(
        session_id="session-1",
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        source_url=f"https://arxiv.org/abs/{paper_id}",
        pipeline_stage="chunk_and_index",
        method_extraction_json={"method_name": "Method"},
        benchmarks_json=[{"task": "task", "metric": "metric", "value": 1.0}],
        readiness_json={"maturity_level": "experimental"},
        full_markdown_report="# Report",
    )


def test_dump_workspaces_jsonl_writes_one_json_object_per_line():
    text = dump_workspaces_jsonl([_workspace("1706.03762"), _workspace("2005.11401")])

    lines = text.splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["paper_id"] == "1706.03762"
    assert json.loads(lines[1])["paper_id"] == "2005.11401"
    assert text.endswith("\n")


def test_export_workspaces_for_session_writes_all_workspaces(tmp_path):
    repository = FakeWorkspaceRepository([_workspace("1706.03762"), _workspace("2005.11401")])
    output = tmp_path / "workspaces.jsonl"

    selected = export_workspaces_for_session(
        repository=repository,
        session_id="session-1",
        output_path=output,
    )

    assert repository.session_ids == ["session-1"]
    assert [workspace.paper_id for workspace in selected] == ["1706.03762", "2005.11401"]
    assert [json.loads(line)["paper_id"] for line in output.read_text().splitlines()] == [
        "1706.03762",
        "2005.11401",
    ]


def test_export_workspaces_for_session_filters_and_orders_requested_papers(tmp_path):
    repository = FakeWorkspaceRepository([_workspace("1706.03762"), _workspace("2005.11401")])
    output = tmp_path / "workspaces.jsonl"

    selected = export_workspaces_for_session(
        repository=repository,
        session_id="session-1",
        output_path=output,
        paper_ids=["2005.11401", "1706.03762"],
    )

    assert [workspace.paper_id for workspace in selected] == ["2005.11401", "1706.03762"]
    assert [json.loads(line)["paper_id"] for line in output.read_text().splitlines()] == [
        "2005.11401",
        "1706.03762",
    ]


def test_export_workspaces_for_session_rejects_missing_requested_paper(tmp_path):
    repository = FakeWorkspaceRepository([_workspace("1706.03762")])

    with pytest.raises(WorkspaceExportError, match="not found: 2005.11401"):
        export_workspaces_for_session(
            repository=repository,
            session_id="session-1",
            output_path=tmp_path / "workspaces.jsonl",
            paper_ids=["2005.11401"],
        )


def test_export_workspaces_for_session_rejects_empty_export(tmp_path):
    repository = FakeWorkspaceRepository([])

    with pytest.raises(WorkspaceExportError, match="No paper workspaces"):
        export_workspaces_for_session(
            repository=repository,
            session_id="session-1",
            output_path=tmp_path / "workspaces.jsonl",
        )

