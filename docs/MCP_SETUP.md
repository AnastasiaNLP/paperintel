# PaperIntel MCP Server

PaperIntel includes a local MCP server for Claude Desktop and other MCP clients.
It exposes the same application surface as the REST API through MCP tools.

## Prerequisites

- PaperIntel dependencies installed in `.venv`
- Postgres and Qdrant running
- Anthropic and OpenAI credentials configured in your environment or `.env`
- Claude Desktop or another MCP client

Start local services:

```bash
docker compose up -d postgres qdrant
```

## Tools

The MCP server exposes seven tools:

- `create_session(persona)` creates a PaperIntel session and returns a session ID.
- `analyze_paper(session_id, paper_url)` analyzes an arXiv or PDF URL. This is synchronous and can take about one minute.
- `ask_paper(session_id, question)` asks a question about papers analyzed in that session.
- `discover_papers(session_id, topic)` searches for candidate papers on arXiv and returns a shortlist.
- `select_papers(session_id, selection)` selects papers from the current discovery shortlist by display number.
- `analyze_selected_papers(session_id)` analyzes papers selected from the discovery shortlist.
- `get_session(session_id)` returns persona, phase, and active paper IDs.

The server does not keep an implicit current session. Pass the `session_id` returned by `create_session` to later tool calls.

## Run Locally

From the repository root:

```bash
.venv/bin/python -m mcp_server.server
```

The server uses the STDIO MCP transport. Do not add `print()` logging to stdout in this process, because stdout carries JSON-RPC messages. Use stderr or Python logging instead.

## Claude Desktop Configuration

Add this to `claude_desktop_config.json`, replacing paths with absolute paths for your machine:

```json
{
  "mcpServers": {
    "paperintel": {
      "command": "/absolute/path/to/paperintel/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/paperintel"
    }
  }
}
```

Restart Claude Desktop after editing the config.

## Example Flow

Ask Claude:

```text
Create a PaperIntel session as engineer and analyze https://arxiv.org/abs/1706.03762.
```

Claude should call:

1. `create_session`
2. `analyze_paper`

Then ask:

```text
What is the main contribution of that paper?
```

Claude should call `ask_paper` with the same `session_id`.

Discovery flow:

```text
Find recent papers about retrieval augmented generation.
```

Claude should call `discover_papers` with the same `session_id`. After the
shortlist appears, choose papers by display number:

```text
Select papers 1 and 3.
```

Claude should call `select_papers`.

To analyze the selected papers:

```text
Analyze the selected papers.
```

Claude should call `analyze_selected_papers`. After analysis finishes, ask
questions with `ask_paper`.

The full discovery workflow is:

```text
create_session -> discover_papers -> select_papers -> analyze_selected_papers -> ask_paper
```

## Troubleshooting

- Tool does not appear: restart Claude Desktop and verify the config path is absolute.
- Analysis or discovery takes a long time: this is expected; both are synchronous in the current MCP adapter.
- Server exits immediately: run `.venv/bin/python -m mcp_server.server` from the repository root to see import/configuration errors.
- JSON-RPC parse errors: ensure the MCP server is not writing logs to stdout.
