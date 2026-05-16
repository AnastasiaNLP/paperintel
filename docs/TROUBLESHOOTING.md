# Troubleshooting

## API is not reachable

Symptom:

```text
Connection refused
```

Start the REST API:

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/uvicorn api.rest.main:app --host 127.0.0.1 --port 8000
```

## Health endpoint returns 503

Check local services:

```bash
docker compose ps
docker compose up -d postgres qdrant
```

Run migrations:

```bash
.venv/bin/python -m alembic upgrade head
```

Note: `alembic.ini` currently contains the default local docker-compose database
URL. If you changed Postgres settings in `.env`, update `alembic.ini` to match.

Verify `.env` has API keys and local service URLs:

```text
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
POSTGRES_URL=postgresql://paperintel:dev_password@localhost:5432/paperintel
QDRANT_URL=http://localhost:6333
```

## Analysis takes a long time

This is expected. Paper analysis is synchronous in the current REST and MCP
adapters and can take about a minute for a typical arXiv paper.

## Asking questions returns weak or insufficient evidence

Make sure the paper was successfully indexed. Check the session:

```bash
curl http://127.0.0.1:8000/sessions/<SESSION_ID>
```

`active_paper_ids` should include the paper ID. If it is empty, indexing did not
complete successfully and the paper is not available for retrieval-backed QA.

## LangSmith traces appear during local tests

Disable tracing for local test runs:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -m 'not live'
```

## MCP tool does not appear in Claude Desktop

- Use absolute paths in `claude_desktop_config.json`.
- Restart Claude Desktop after editing the config.
- Run the server manually from the repository root to catch import errors:

```bash
.venv/bin/python -m mcp_server.server
```

The MCP server uses STDIO. Do not add `print()` statements to stdout in this
process.

## Database tests are skipped

This is expected unless `PAPERINTEL_TEST_DATABASE_URL` is set:

```bash
export PAPERINTEL_TEST_DATABASE_URL=postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel
```

Then run the DB-marked tests explicitly.
