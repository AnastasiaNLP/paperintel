import asyncio

import pytest

import mcp_server.tools as tool_module
from mcp_server.tools import (
    analyze_paper_tool,
    analyze_selected_papers_tool,
    ask_paper_tool,
    create_session_tool,
    discover_papers_tool,
    format_answer_result,
    format_discovery_result,
    get_session_tool,
    select_papers_tool,
)
from models.retrieval import CitationRef
from models.session import HandlerResult, Session


class FakeService:
    def __init__(self) -> None:
        self.sessions = {
            "session-1": Session(
                id="session-1",
                persona="engineer",
                phase="qa",
                active_paper_ids=["1706.03762"],
            )
        }
        self.create_calls = []
        self.analyze_calls = []
        self.ask_calls = []
        self.discover_calls = []
        self.select_calls = []
        self.analyze_selected_calls = []

    def create_session(self, *, persona="engineer", original_query=None):
        self.create_calls.append({"persona": persona, "original_query": original_query})
        return Session(id="created-session", persona=persona)

    def analyze_paper(self, session_id, paper_url):
        self.analyze_calls.append((session_id, paper_url))
        return HandlerResult(
            session_id=session_id,
            response_text="Analysis complete.",
            phase="qa",
            referenced_paper_ids=["1706.03762"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def ask_question(self, session_id, question):
        self.ask_calls.append((session_id, question))
        return HandlerResult(
            session_id=session_id,
            response_text="The Transformer replaces recurrence with attention.",
            phase="qa",
            intent="qa_factual",
            citations=[
                CitationRef(
                    paper_id="1706.03762",
                    chunk_id="1706.03762:chunk:1",
                    page_start=1,
                    page_end=1,
                )
            ],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def discover_papers(self, session_id, topic):
        self.discover_calls.append((session_id, topic))
        return HandlerResult(
            session_id=session_id,
            response_text="Here are candidate papers. Reply with numbers.",
            phase="selection",
            intent="discover",
            discovery_topic="agent memory",
            discovery_candidate_count=3,
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def select_papers(self, session_id, selection):
        self.select_calls.append((session_id, selection))
        return HandlerResult(
            session_id=session_id,
            response_text="Selected papers 1 and 3.",
            phase="idle",
            intent="select_papers",
            selected_candidate_ids=["candidate-1", "candidate-3"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_selected_papers(self, session_id):
        self.analyze_selected_calls.append(session_id)
        return HandlerResult(
            session_id=session_id,
            response_text="Selected analysis complete.",
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=["2605.1", "2605.3"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def get_session(self, session_id):
        return self.sessions[session_id]


class ExplodingService(FakeService):
    def ask_question(self, session_id, question):
        raise RuntimeError("internal details should not leak")

    def analyze_selected_papers(self, session_id):
        raise RuntimeError("internal details should not leak")


@pytest.fixture(autouse=True)
def run_sync_inline(monkeypatch):
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(tool_module, "_run_sync", inline)


def test_create_session_tool_returns_session_id():
    service = FakeService()

    text = asyncio.run(create_session_tool(service, persona="researcher"))

    assert "Session ID: created-session" in text
    assert "Persona: researcher" in text
    assert "discover_papers" in text
    assert service.create_calls == [
        {"persona": "researcher", "original_query": None}
    ]


def test_create_session_rejects_invalid_persona():
    with pytest.raises(ValueError):
        asyncio.run(create_session_tool(FakeService(), persona="manager"))


def test_analyze_paper_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        analyze_paper_tool(
            service,
            session_id="session-1",
            paper_url="https://arxiv.org/abs/1706.03762",
        )
    )

    assert "Paper analysis completed." in text
    assert "1706.03762" in text
    assert service.analyze_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]


def test_analyze_paper_rejects_non_url():
    with pytest.raises(ValueError):
        asyncio.run(
            analyze_paper_tool(
                FakeService(),
                session_id="session-1",
                paper_url="arxiv 1706.03762",
            )
        )


def test_ask_paper_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        ask_paper_tool(
            service,
            session_id="session-1",
            question="What is the contribution?",
        )
    )

    assert "The Transformer replaces recurrence" in text
    assert "Sources:" in text
    assert service.ask_calls == [("session-1", "What is the contribution?")]


def test_ask_paper_rejects_empty_question():
    with pytest.raises(ValueError):
        asyncio.run(ask_paper_tool(FakeService(), session_id="session-1", question=""))


def test_ask_paper_rejects_too_long_question():
    with pytest.raises(ValueError):
        asyncio.run(
            ask_paper_tool(
                FakeService(),
                session_id="session-1",
                question="x" * 2001,
            )
        )


def test_discover_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        discover_papers_tool(
            service,
            session_id="session-1",
            topic="Find papers about agent memory",
        )
    )

    assert "Here are candidate papers" in text
    assert "Candidates found: 3" in text
    assert "Session phase: selection" in text
    assert service.discover_calls == [
        ("session-1", "Find papers about agent memory")
    ]


def test_discover_papers_rejects_empty_topic():
    with pytest.raises(ValueError):
        asyncio.run(
            discover_papers_tool(FakeService(), session_id="session-1", topic="")
        )


def test_select_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        select_papers_tool(
            service,
            session_id="session-1",
            selection="use 1 and 3",
        )
    )

    assert "Selected papers 1 and 3" in text
    assert "candidate-1" in text
    assert service.select_calls == [("session-1", "use 1 and 3")]


def test_select_papers_rejects_empty_selection():
    with pytest.raises(ValueError):
        asyncio.run(
            select_papers_tool(FakeService(), session_id="session-1", selection="")
        )


def test_analyze_selected_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        analyze_selected_papers_tool(service, session_id="session-1")
    )

    assert "Paper analysis completed." in text
    assert "Selected analysis complete." in text
    assert "- 2605.1" in text
    assert service.analyze_selected_calls == ["session-1"]


def test_analyze_selected_papers_rejects_empty_session_id():
    with pytest.raises(ValueError):
        asyncio.run(analyze_selected_papers_tool(FakeService(), session_id=""))


def test_analyze_selected_papers_handles_service_exception_safely():
    text = asyncio.run(
        analyze_selected_papers_tool(ExplodingService(), session_id="session-1")
    )

    assert "could not analyze the selected papers safely" in text
    assert "internal details" not in text


def test_get_session_tool_returns_state():
    text = asyncio.run(get_session_tool(FakeService(), session_id="session-1"))

    assert "Session: session-1" in text
    assert "Persona: engineer" in text
    assert "- 1706.03762" in text


def test_format_answer_result_includes_citations():
    result = HandlerResult(
        session_id="session-1",
        response_text="Answer.",
        phase="qa",
        citations=[
            CitationRef(
                paper_id="1706.03762",
                chunk_id="1706.03762:chunk:14",
                page_start=8,
                page_end=9,
            )
        ],
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )

    text = format_answer_result(result)

    assert "Sources:" in text
    assert "1706.03762, pages 8-9, chunk 1706.03762:chunk:14" in text


def test_format_discovery_result_includes_topic_and_count():
    result = HandlerResult(
        session_id="session-1",
        response_text="Choose papers.",
        phase="selection",
        discovery_topic="agent memory",
        discovery_candidate_count=5,
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )

    text = format_discovery_result(result)

    assert "Topic: agent memory" in text
    assert "Candidates found: 5" in text
    assert "select_papers" in text


def test_tool_handles_service_exception_safely():
    text = asyncio.run(
        ask_paper_tool(
            ExplodingService(),
            session_id="session-1",
            question="What is the contribution?",
        )
    )

    assert "could not answer the question safely" in text
    assert "internal details" not in text
