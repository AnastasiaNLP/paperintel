from mcp.server.fastmcp import FastMCP

from api.app_factory import create_paperintel_service
from services.paperintel_service import PaperIntelService


def create_mcp_server(*, service: PaperIntelService | None = None) -> FastMCP:
    service = service or create_paperintel_service()
    mcp = FastMCP("paperintel")

    @mcp.tool()
    async def create_session(persona: str = "engineer") -> str:
        """Create a PaperIntel session and return a session_id for later calls."""
        from mcp_server.tools import create_session_tool

        return await create_session_tool(service, persona=persona)

    @mcp.tool()
    async def analyze_paper(session_id: str, paper_url: str) -> str:
        """Analyze an arXiv or PDF paper URL. This can take about one minute."""
        from mcp_server.tools import analyze_paper_tool

        return await analyze_paper_tool(
            service,
            session_id=session_id,
            paper_url=paper_url,
        )

    @mcp.tool()
    async def ask_paper(session_id: str, question: str) -> str:
        """Ask a question about papers already analyzed in this session."""
        from mcp_server.tools import ask_paper_tool

        return await ask_paper_tool(
            service,
            session_id=session_id,
            question=question,
        )

    @mcp.tool()
    async def discover_papers(session_id: str, topic: str) -> str:
        """Find relevant papers for a research topic and present a shortlist."""
        from mcp_server.tools import discover_papers_tool

        return await discover_papers_tool(
            service,
            session_id=session_id,
            topic=topic,
        )

    @mcp.tool()
    async def select_papers(session_id: str, selection: str) -> str:
        """Select papers from the current discovery shortlist by display number."""
        from mcp_server.tools import select_papers_tool

        return await select_papers_tool(
            service,
            session_id=session_id,
            selection=selection,
        )

    @mcp.tool()
    async def analyze_selected_papers(session_id: str) -> str:
        """Analyze papers previously selected from a discovery shortlist."""
        from mcp_server.tools import analyze_selected_papers_tool

        return await analyze_selected_papers_tool(service, session_id=session_id)

    @mcp.tool()
    async def get_session(session_id: str) -> str:
        """Get persona, phase, and active papers for a PaperIntel session."""
        from mcp_server.tools import get_session_tool

        return await get_session_tool(service, session_id=session_id)

    return mcp


def main() -> None:
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
