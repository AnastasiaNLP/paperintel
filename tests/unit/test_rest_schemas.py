import pytest
from pydantic import ValidationError

from api.rest.schemas import (
    AnalyzeRequest,
    AskRequest,
    DiscoverRequest,
    MessageResponse,
    SelectPapersRequest,
)
from models.session import HandlerResult


def test_analyze_request_requires_valid_url():
    payload = AnalyzeRequest(paper_url="https://arxiv.org/abs/1706.03762")

    assert str(payload.paper_url).startswith("https://arxiv.org/abs/1706.03762")

    with pytest.raises(ValidationError):
        AnalyzeRequest(paper_url="arxiv 1706.03762")


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_rejects_question_over_max_length():
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 2001)


def test_discover_request_rejects_empty_topic():
    with pytest.raises(ValidationError):
        DiscoverRequest(topic="")


def test_select_papers_request_rejects_empty_selection():
    with pytest.raises(ValidationError):
        SelectPapersRequest(selection="")


def test_message_response_excludes_internal_handler_fields():
    result = HandlerResult(
        session_id="session-1",
        response_text="Answer",
        phase="qa",
        intent="qa_factual",
        referenced_paper_ids=["1706.03762"],
        needs_analysis=False,
        user_turn_id="turn-user",
        assistant_turn_id="turn-assistant",
    )

    payload = MessageResponse.from_handler_result(result).model_dump(mode="json")

    assert payload["response_text"] == "Answer"
    assert payload["referenced_paper_ids"] == ["1706.03762"]
    assert "agent_runs" not in payload
    assert "errors" not in payload
    assert "raw" not in payload


def test_message_response_includes_discovery_metadata():
    result = HandlerResult(
        session_id="session-1",
        response_text="Choose papers.",
        phase="selection",
        intent="discover",
        discovery_topic="agent memory",
        discovery_candidate_count=4,
        selected_candidate_ids=["candidate-1"],
        user_turn_id="turn-user",
        assistant_turn_id="turn-assistant",
    )

    payload = MessageResponse.from_handler_result(result).model_dump(mode="json")

    assert payload["discovery_topic"] == "agent memory"
    assert payload["discovery_candidate_count"] == 4
    assert payload["selected_candidate_ids"] == ["candidate-1"]


def test_message_response_includes_comparison_markdown():
    result = HandlerResult(
        session_id="session-1",
        response_text="Analysis complete.",
        phase="qa",
        intent="analyze_paper",
        comparison_markdown="# Paper Comparison\n\nA vs B.",
        user_turn_id="turn-user",
        assistant_turn_id="turn-assistant",
    )

    payload = MessageResponse.from_handler_result(result).model_dump(mode="json")

    assert payload["comparison_markdown"] == "# Paper Comparison\n\nA vs B."
