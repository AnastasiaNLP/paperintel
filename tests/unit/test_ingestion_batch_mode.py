import importlib
import sys
import types


def _load_ingestion_with_stubs():
    settings_module = types.ModuleType("config.settings")
    settings_module.settings = types.SimpleNamespace()
    sys.modules["config.settings"] = settings_module

    arxiv_client = types.ModuleType("tools.arxiv_client")
    arxiv_client.download_pdf = lambda arxiv_id: "/tmp/fake.pdf"
    arxiv_client.get_metadata = lambda arxiv_id: types.SimpleNamespace(
        title="Stub title",
        authors=["Stub author"],
        arxiv_id=arxiv_id,
        published_date="2026-01-01",
        abstract="Stub abstract",
        categories=["cs.AI"],
    )
    sys.modules["tools.arxiv_client"] = arxiv_client

    pdf_parser = types.ModuleType("tools.pdf_parser")
    pdf_parser.parse_pdf = lambda path: {
        "raw_text": "arXiv: 2501.12948\npaper text",
        "text_by_page": {1: "page one"},
        "metadata": {"title": "Stub title"},
        "arxiv_id": "2501.12948",
    }
    sys.modules["tools.pdf_parser"] = pdf_parser

    semantic_scholar = types.ModuleType("tools.semantic_scholar_client")
    semantic_scholar.get_paper = lambda arxiv_id: {"citation_count": 10}
    sys.modules["tools.semantic_scholar_client"] = semantic_scholar

    sys.modules.pop("agents.ingestion", None)
    return importlib.import_module("agents.ingestion")


def test_single_url_still_uses_input_value():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "current_paper_index": 0,
        "total_papers": 1,
    }

    assert ingestion._resolve_current_url(state) == "https://arxiv.org/abs/2501.12948"


def test_batch_mode_uses_batch_urls_current_index_zero():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "ignored-in-batch",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 0,
        "total_papers": 2,
    }

    assert ingestion._resolve_current_url(state) == "https://arxiv.org/abs/2501.12948"


def test_batch_mode_uses_batch_urls_current_index_one():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "ignored-in-batch",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 1,
        "total_papers": 2,
    }

    assert ingestion._resolve_current_url(state) == "https://arxiv.org/abs/2305.14314"


def test_validate_input_fails_on_invalid_batch_index():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "ignored-in-batch",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 5,
        "total_papers": 2,
    }

    assert ingestion._validate_input(state) == "batch_urls index 5 out of range"


def test_validate_input_fails_on_total_papers_mismatch():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "ignored-in-batch",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 0,
        "total_papers": 3,
    }

    assert "total_papers mismatch" in ingestion._validate_input(state)


def test_validate_input_fails_when_batch_mode_is_not_url():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "pdf",
        "input_value": "/tmp/paper.pdf",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 0,
        "total_papers": 2,
    }

    assert ingestion._validate_input(state) == (
        "batch mode currently supports only input_type='url'"
    )


def test_ingestion_agent_returns_fatal_error_for_invalid_batch_setup():
    ingestion = _load_ingestion_with_stubs()
    state = {
        "input_type": "url",
        "input_value": "ignored-in-batch",
        "batch_urls": [
            "https://arxiv.org/abs/2501.12948",
            "https://arxiv.org/abs/2305.14314",
        ],
        "current_paper_index": 5,
        "total_papers": 2,
    }

    result = ingestion.ingestion_agent(state)

    assert result["processing_stage"] == "failed"
    assert result["paper_failed"] is False
    assert result["failed_node"] == "ingestion"

