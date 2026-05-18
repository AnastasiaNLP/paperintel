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

## Discovery returns no papers or live discovery tests skip

Discovery uses the public arXiv API. If arXiv rate-limits requests, PaperIntel
records warnings such as:

```text
Search query failed (HTTP 429): retrieval augmented generation
```

The live discovery tests skip when every arXiv search query is rate-limited.
Wait a few minutes and retry. Partial rate limits are tolerated: if another
query returns candidates, the workflow continues.

## Analyze selected papers returns no active papers

Selected-paper analysis depends on arXiv metadata and PDF retrieval for the
chosen candidate. If arXiv metadata retrieval fails, the analysis phase is
marked `failed` and the paper is not added to `active_paper_ids`.

Try again later, or select a different candidate from the discovery shortlist.

## Asking questions returns weak or insufficient evidence

Make sure the paper was successfully indexed. Check the session:

```bash
curl http://127.0.0.1:8000/sessions/<SESSION_ID>
```

`active_paper_ids` should include the paper ID. If it is empty, indexing did not
complete successfully and the paper is not available for retrieval-backed QA.

## Synthesis returns `no_active_papers`

`/synthesize` and the MCP `synthesize_papers` tool require at least one active
paper in the session. Analyze a paper URL or analyze selected discovery
candidates first, then retry synthesis.

## No comparison report appears after analyzing selected papers

Batch comparison is generated only when multiple selected papers are analyzed
together and the analysis graph completes successfully for at least two papers.
If you selected only one paper, or one of the selected papers failed during
metadata/PDF retrieval, the session can still support QA but may not include a
batch comparison artifact.

For on-demand comparison after papers are active, use `/synthesize` or the MCP
`synthesize_papers` tool. That path uses retrieval-backed QA with citations
rather than the batch comparator.

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
