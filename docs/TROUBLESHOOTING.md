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

## Local PDF analysis fails

REST local PDF analysis uses multipart upload at
`POST /sessions/{id}/analyze-pdf`. It does not accept server-local file
paths. MCP local PDF analysis uses `analyze_pdf` and reads a trusted path
from the MCP server machine.

Common failures:

- `pdf_too_large`: the upload is larger than 50 MB. Use a smaller PDF or
  split the document before analysis.
- `unsupported_media_type`: the upload is not `application/pdf` or does
  not start with `%PDF-`.
- `invalid_pdf_input`: the service rejected the local path or file after
  validation. For MCP, verify the path is absolute, exists on the MCP
  server machine, is a file, is under 50 MB, and starts with `%PDF-`.

PDF bytes are temporary. PaperIntel deletes REST upload temp files after
analysis and persists only workspaces, chunks, and artifacts. Async PDF
upload jobs are deferred until shared PDF/object storage exists.

## Analyze selected papers returns no active papers

Selected-paper analysis depends on retrieving usable PDF/text for the chosen
candidate. arXiv metadata failure alone should not fail analysis: PaperIntel
first checks the Postgres metadata cache, then the arXiv API, and can continue
with PDF-derived fallback metadata when the PDF is available.

If no active papers are produced, inspect the job/session errors for PDF
download, PDF parsing, indexing, or provider failures. Retry later if the public
arXiv PDF service or embedding provider is unavailable, or select a different
candidate from the discovery shortlist.

## arXiv or Semantic Scholar rate limits

PaperIntel has process-local rate limiters and in-memory circuit breakers for
arXiv and Semantic Scholar. arXiv metadata is cached in Postgres, and Semantic
Scholar enrichment is optional. If a breaker opens, requests fail fast until the
half-open timeout instead of repeatedly calling the unhealthy upstream service.

Current limitations:

- The limiter and breaker are process-local. Multiple REST/worker processes can
  still exceed upstream limits collectively.
- arXiv metadata failures can degrade to PDF fallback metadata, but PDF download
  and parsing must still succeed for URL analysis to continue.
- Semantic Scholar failures remove citation enrichment but should not fail
  analysis.

See `docs/RESILIENCE.md` for details.

## Asking questions returns weak or insufficient evidence

Make sure the paper was successfully indexed. Check the session:

```bash
curl http://127.0.0.1:8000/sessions/<SESSION_ID>
```

`active_paper_ids` should include the paper ID. If it is empty, indexing did not
complete successfully and the paper is not available for retrieval-backed QA.

## Synthesis returns `no_active_papers`

`/synthesize` and the MCP `synthesize_papers` tool require at least two
distinct ready paper workspaces. If the session has no active papers, analyze
paper URLs or selected discovery candidates first. If there is only one active
paper, analyze another paper before retrying synthesis.

## No comparison report appears after analyzing selected papers

Batch comparison is generated only when multiple selected papers are analyzed
together and the analysis graph completes successfully for at least two papers.
If you selected only one paper, or one of the selected papers failed during
metadata/PDF retrieval, the session can still support QA but may not include a
batch comparison artifact.

For on-demand comparison after papers are active, use `POST /sessions/{id}/compare`
or the MCP `compare_papers` tool. That request-driven path loads durable
workspaces and creates a new `ComparisonArtifact` with
`producer="comparison_analyst"`. Use `/synthesize` or MCP `synthesize_papers`
when you want a persona-aware recommendation rather than a persisted comparison
artifact.

## `GET /sessions/{id}/comparison` returns `comparison_not_found`

This is expected until either multi-paper batch analysis or request-driven
`/compare` has produced a comparison artifact. Analyze at least two papers and
then call `POST /sessions/{id}/compare`, or analyze selected papers together
with `/analyze-selected`, then retry the comparison endpoint or MCP
`get_latest_comparison` tool.

Single-paper analysis still persists a paper workspace. Use
`/sessions/{id}/workspaces` or MCP `list_paper_workspaces` to inspect saved
per-paper artifacts.

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
