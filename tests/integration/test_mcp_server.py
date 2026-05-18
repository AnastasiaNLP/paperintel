import asyncio

import pytest

from models.session import HandlerResult, Session


class FakeService:
    def create_session(self, *, persona="engineer", original_query=None):
        return Session(id="session-1", persona=persona)

    def analyze_paper(self, session_id, paper_url):
        return HandlerResult(
            session_id=session_id,
            response_text="Analysis complete.",
            phase="qa",
            referenced_paper_ids=["1706.03762"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def ask_question(self, session_id, question):
        return HandlerResult(
            session_id=session_id,
            response_text="Answer.",
            phase="qa",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def discover_papers(self, session_id, topic):
        return HandlerResult(
            session_id=session_id,
            response_text="Candidates.",
            phase="selection",
            intent="discover",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def select_papers(self, session_id, selection):
        return HandlerResult(
            session_id=session_id,
            response_text="Selected.",
            phase="idle",
            intent="select_papers",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_selected_papers(self, session_id):
        return HandlerResult(
            session_id=session_id,
            response_text="Selected analysis complete.",
            phase="qa",
            intent="analyze_paper",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def synthesize_papers(self, session_id, prompt=None):
        return HandlerResult(
            session_id=session_id,
            response_text="Synthesis.",
            phase="qa",
            intent="qa_comparison",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def get_session(self, session_id):
        return Session(id=session_id, phase="qa", active_paper_ids=["1706.03762"])


def test_mcp_server_builds_with_four_tools():
    pytest.importorskip("mcp")
    from mcp_server.server import create_mcp_server

    server = create_mcp_server(service=FakeService())

    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert {
        "create_session",
        "analyze_paper",
        "ask_paper",
        "discover_papers",
        "select_papers",
        "analyze_selected_papers",
        "synthesize_papers",
        "get_session",
    }.issubset(names)


def test_mcp_server_has_stdio_entrypoint():
    pytest.importorskip("mcp")
    from mcp_server.server import main

    assert callable(main)
