# Live Tests

Live tests exercise PaperIntel against real local services and, for selected
flows, live LLM providers. They are opt-in and are not part of the default test
suite.

## CA Live Smoke

`tests/live/test_ca_live_smoke.py` verifies the comparison/synthesis public
surfaces end-to-end on the real stack:

- Session A: analyze two local PDFs, then synthesize without comparison context.
- Session B: analyze two local PDFs, compare through REST, reload latest
  comparison through REST, then synthesize with comparison context.
- MCP tools: `compare_papers` and `synthesize_papers` return successful readable
  output.
- Postgres verification: workspaces, comparison artifact count, AgentRun status,
  `policy_applied`, and output refs.
- Cleanup: temporary Qdrant collection and Postgres foundation tables.

The test is skipped unless `PAPERINTEL_RUN_LIVE_CA_SMOKE=1` is set.

## Prerequisites

Start local services:

```bash
docker compose up -d postgres qdrant
```

Required environment variables, usually loaded from `.env`:

```text
PAPERINTEL_TEST_DATABASE_URL
PAPERINTEL_QDRANT_TEST_URL
ANTHROPIC_API_KEY
OPENAI_API_KEY
```

Required local PDFs:

```text
~/Desktop/pdfs/1706.03762.pdf
~/Desktop/pdfs/2005.11401.pdf
```

The test uses local PDFs and calls `analyze_pdf(..., skip_arxiv_metadata_fetch=True)`
so it does not depend on arXiv metadata availability for the smoke path.

## Run Without LangSmith Trace

Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid unrelated ROS pytest plugins from
loading into the Python 3.11 environment.

```bash
PAPERINTEL_RUN_LIVE_CA_SMOKE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_ca_live_smoke.py
```

## Run With LangSmith Trace

Tracing is off by default inside the test. Enable it explicitly:

```bash
PAPERINTEL_RUN_LIVE_CA_SMOKE=1 \
PAPERINTEL_LIVE_TRACE=1 \
LANGCHAIN_TRACING_V2=true \
LANGSMITH_TRACING=true \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_ca_live_smoke.py
```

Make sure `.env` contains `LANGSMITH_API_KEY`. Set `LANGCHAIN_PROJECT` if you
want the run grouped under a specific project.

## Expected Success Markers

A successful run should include markers like:

```text
LIVE_CA_SESSION_A=<uuid>
LIVE_CA_SESSION_B=<uuid>
LIVE_CA_AGENT_RUN=comparison_analyst:<uuid>:completed:comparison_report
LIVE_CA_AGENT_RUN=synthesis_agent:<uuid>:completed:synthesis_report
LIVE_CA_COMPARISON_COUNT_<session_a>=0
LIVE_CA_COMPARISON_COUNT_<session_b>=1
LIVE_CA_QDRANT_CLEANUP=success
LIVE_CA_POSTGRES_CLEANUP=success
1 passed
```

`fallback_used` is treated as a test failure. The smoke is intended to verify the
happy path, not only deterministic fallback behavior.

## Common Failures

ROS pytest plugin error:

```text
AttributeError: module 'asyncio' has no attribute 'coroutine'
```

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Missing local PDFs:

```text
Local PDFs are required for CA live smoke
```

Place the expected PDFs in `~/Desktop/pdfs` or update the test fixture before
running.

Cleanup failure:

```text
LIVE_CA_QDRANT_CLEANUP=failed:...
```

The test uses a per-run Qdrant collection named `paper_chunks_ca_live_<run_id>`.
If automatic cleanup fails, delete that collection manually and inspect Postgres
test tables before re-running.
