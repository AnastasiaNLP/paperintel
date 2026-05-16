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

The MCP server exposes four tools:

- `create_session(persona)` creates a PaperIntel session and returns a session ID.
- `analyze_paper(session_id, paper_url)` analyzes an arXiv or PDF URL. This is synchronous and can take about one minute.
- `ask_paper(session_id, question)` asks a question about papers analyzed in that session.
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

## Troubleshooting

- Tool does not appear: restart Claude Desktop and verify the config path is absolute.
- Analysis takes a long time: this is expected; paper analysis is synchronous in the current MCP adapter.
- Server exits immediately: run `.venv/bin/python -m mcp_server.server` from the repository root to see import/configuration errors.
- JSON-RPC parse errors: ensure the MCP server is not writing logs to stdout.
