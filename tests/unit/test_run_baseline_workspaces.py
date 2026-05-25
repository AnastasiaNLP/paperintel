from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.run_baseline_workspaces import BaselinePaper, load_baseline_papers, run_baseline
from models.artifacts import PaperWorkspace


class FakeArtifactRepository:
    def list_workspaces(self, session_id):
        return [
            PaperWorkspace(
                session_id=session_id,
                paper_id="paper-1",
                title="Paper 1",
                source_url="https://arxiv.org/abs/1234.56789",
                pipeline_stage="completed",
            )
        ]


class FakeHandler:
    def __init__(self):
        self.analysis_metadata_fallback_by_arxiv_id = {}


class FakeService:
    def __init__(self):
        self.handler = FakeHandler()
        self.artifact_repository = FakeArtifactRepository()
        self.analyzed_pdfs = []

    def create_session(self, *, persona, original_query):
        return SimpleNamespace(id="session-1")

    def analyze_paper(self, session_id, paper_url):
        return SimpleNamespace(
            phase="qa",
            errors=[SimpleNamespace(message="warning only")],
            error=SimpleNamespace(message="warning only"),
        )

    def analyze_pdf(
        self,
        session_id,
        pdf_path,
        *,
        paper_id=None,
        skip_arxiv_metadata_fetch=False,
    ):
        self.analyzed_pdfs.append(
            {
                "session_id": session_id,
                "pdf_path": pdf_path,
                "paper_id": paper_id,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
            }
        )
        return SimpleNamespace(
            phase="qa",
            errors=[],
            error=None,
        )


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


def test_load_baseline_papers_attaches_local_pdf_paths(tmp_path):
    pdf_path = tmp_path / "1706.03762.pdf"
    pdf_path.write_bytes(b"%PDF")

    papers = load_baseline_papers(
        "golden_dataset/paperintel_30_v0_1.jsonl",
        selected_paper_ids=["1706.03762"],
        pdf_dir=tmp_path,
        require_local_pdfs=True,
    )

    assert papers[0].local_pdf_path == pdf_path


def test_load_baseline_papers_can_require_local_pdfs(tmp_path):
    with pytest.raises(ValueError, match="Missing local PDFs"):
        load_baseline_papers(
            "golden_dataset/paperintel_30_v0_1.jsonl",
            selected_paper_ids=["1706.03762"],
            pdf_dir=tmp_path,
            require_local_pdfs=True,
        )


def test_load_baseline_papers_rejects_unknown_requested_id():
    with pytest.raises(ValueError, match="not in the golden dataset"):
        load_baseline_papers(
            "golden_dataset/paperintel_30_v0_1.jsonl",
            selected_paper_ids=["missing"],
        )


def test_run_baseline_treats_warnings_as_success(tmp_path):
    result = run_baseline(
        service=FakeService(),
        papers=[
            BaselinePaper(
                paper_id="paper-1",
                source_url="https://arxiv.org/abs/1234.56789",
                title="Paper 1",
                metadata_fallback={
                    "title": "Paper 1",
                    "authors": [],
                    "arxiv_id": "paper-1",
                    "published_date": "",
                    "abstract": "",
                    "categories": [],
                },
            )
        ],
        output_path=tmp_path / "workspaces.jsonl",
        persona="engineer",
        resume_session_id=None,
        skip_existing=False,
        sleep_seconds=0,
        continue_on_error=False,
    )

    assert result.analyzed_count == 1
    assert result.failed_count == 0


def test_run_baseline_uses_local_pdf_with_golden_metadata(tmp_path):
    service = FakeService()
    pdf_path = tmp_path / "paper-1.pdf"
    pdf_path.write_bytes(b"%PDF")

    result = run_baseline(
        service=service,
        papers=[
            BaselinePaper(
                paper_id="paper-1",
                source_url="https://arxiv.org/abs/1234.56789",
                title="Paper 1",
                metadata_fallback={
                    "title": "Paper 1",
                    "authors": [],
                    "arxiv_id": "paper-1",
                    "published_date": "",
                    "abstract": "",
                    "categories": [],
                },
                local_pdf_path=pdf_path,
            )
        ],
        output_path=tmp_path / "workspaces.jsonl",
        persona="engineer",
        resume_session_id=None,
        skip_existing=False,
        sleep_seconds=0,
        continue_on_error=False,
        metadata_source="golden",
    )

    assert result.failed_count == 0
    assert service.analyzed_pdfs == [
        {
            "session_id": "session-1",
            "pdf_path": str(pdf_path),
            "paper_id": "paper-1",
            "skip_arxiv_metadata_fetch": True,
        }
    ]
