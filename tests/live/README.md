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
docker compose up -d postgres qdrant minio
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

## PDF Live Smoke

`tests/live/test_pdf_live_smoke.py` verifies local PDF product surfaces on the
real stack:

- REST multipart upload: `POST /sessions/{id}/analyze-pdf` analyzes an uploaded
  PDF with `skip_arxiv_metadata_fetch=true`.
- MCP local path: `analyze_pdf` analyzes a trusted local PDF path on the MCP
  server machine.
- Persistence: each path produces one ready `PaperWorkspace` with report and
  method artifacts.
- Cleanup: temporary Qdrant collection and Postgres foundation tables.

The test is skipped unless `PAPERINTEL_RUN_LIVE_PDF_SMOKE=1` is set.

Required local PDF:

```text
~/Desktop/pdfs/1706.03762.pdf
```

### Run Without LangSmith Trace

```bash
PAPERINTEL_RUN_LIVE_PDF_SMOKE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_pdf_live_smoke.py
```

### Run With LangSmith Trace

```bash
PAPERINTEL_RUN_LIVE_PDF_SMOKE=1 \
PAPERINTEL_LIVE_TRACE=1 \
LANGCHAIN_TRACING_V2=true \
LANGSMITH_TRACING=true \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_pdf_live_smoke.py
```

### Expected Success Markers

```text
LIVE_PDF_RUN_ID=<run_id>
LIVE_PDF_REST_SESSION_ID=<uuid>
LIVE_PDF_REST_STATUS=200
LIVE_PDF_REST_WORKSPACE_IDS=local-rest-1706
LIVE_PDF_MCP_SESSION_ID=<uuid>
LIVE_PDF_MCP_OUTPUT_CHARS=<n>
LIVE_PDF_MCP_WORKSPACE_IDS=local-mcp-1706
LIVE_PDF_QDRANT_CLEANUP=success
LIVE_PDF_POSTGRES_CLEANUP=success
1 passed
```


## Blob Storage Live Smoke

`tests/live/test_blob_live_smoke.py` verifies durable PDF object storage on the
real stack through the default application factory:

- REST multipart upload persists one PDF through the configured MinIO backend.
- MCP local path analysis uploads the same PDF from a second session.
- Async PDF upload uses the presigned REST lifecycle, enqueues a workflow job,
  and an in-process worker analyzes the stored blob.
- Cancel-before-claim leaves the PDF blob durable and releases the workflow-job
  reference.
- Content-hash deduplication keeps one S3 object and one `blob_artifacts` row.
- Postgres stores session, paper-workspace, and workflow-job references.
- `/health` reports `blob_store=ok`.
- Cleanup removes the temporary Qdrant collection, MinIO bucket, and Postgres
  foundation rows. Blob cleanup retention is also covered by the non-live
  MinIO cleanup smoke in `tests/integration/test_minio_blob_store.py`.

The test is skipped unless `PAPERINTEL_RUN_LIVE_BLOB_SMOKE=1` is set. It also
requires the standard database, Qdrant, and provider variables plus:

```text
PAPERINTEL_TEST_DATABASE_URL
PAPERINTEL_QDRANT_TEST_URL
ANTHROPIC_API_KEY
OPENAI_API_KEY
PAPERINTEL_MINIO_TEST_URL
PAPERINTEL_MINIO_TEST_ACCESS_KEY_ID
PAPERINTEL_MINIO_TEST_SECRET_ACCESS_KEY
```

The run commands below assume those variables are already present in the
environment or loaded from `.env` by `python -m dotenv run`.

Required local PDF:

```text
~/Desktop/pdfs/1706.03762.pdf
```

### Run Without LangSmith Trace

```bash
PAPERINTEL_RUN_LIVE_BLOB_SMOKE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_blob_live_smoke.py
```

### Run With LangSmith Trace

```bash
PAPERINTEL_RUN_LIVE_BLOB_SMOKE=1 \
PAPERINTEL_LIVE_TRACE=1 \
LANGCHAIN_TRACING_V2=true \
LANGSMITH_TRACING=true \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_blob_live_smoke.py
```

### Expected Success Markers

```text
LIVE_BLOB_RUN_ID=<run_id>
LIVE_BLOB_REST_STATUS=200
LIVE_BLOB_MCP_REUSED_ANALYSIS=true
LIVE_BLOB_ASYNC_INITIATE_STATUS=201
LIVE_BLOB_ASYNC_PUT_STATUS=200
LIVE_BLOB_ASYNC_FINALIZE_STATUS=200
LIVE_BLOB_ASYNC_ENQUEUE_STATUS=202
LIVE_BLOB_ASYNC_WORKER_RESULT=<job_id>:succeeded
LIVE_BLOB_ASYNC_CANCEL_STATUS=canceled
LIVE_BLOB_ARTIFACT_COUNT=1
LIVE_BLOB_REFERENCE_COUNT=9
LIVE_BLOB_OBJECT_COUNT=1
LIVE_BLOB_HEALTH_STORE=ok
LIVE_BLOB_QDRANT_CLEANUP=success
LIVE_BLOB_MINIO_CLEANUP=success
LIVE_BLOB_POSTGRES_CLEANUP=success
1 passed
```


## Async Jobs Live Smoke

`tests/live/test_async_jobs_live_smoke.py` verifies the async job path
end-to-end on the real stack:

- REST enqueue: `POST /sessions/{id}/jobs/analyze-paper` creates a queued
  `WorkflowJob`.
- Worker execution: an in-process `WorkflowWorker` claims the Postgres job and
  runs `analyze_paper`.
- REST status: `GET /jobs/{job_id}` returns the succeeded `result_json`.
- MCP status: `get_workflow_job` and `list_workflow_jobs` return readable job
  status output.
- Failure path: an invalid job payload fails deterministically with
  `invalid_job_input`.
- Cancel path: a queued job can be canceled and is not claimed by the worker.
- Cleanup: temporary Qdrant collection and Postgres foundation tables.

The test is skipped unless `PAPERINTEL_RUN_ASYNC_JOBS_LIVE=1` is set.

This smoke currently uses one arXiv URL job (`https://arxiv.org/abs/1706.03762`)
and therefore can exercise the arXiv metadata cache, process-local limiter,
circuit breaker, and PDF fallback path when upstream services are slow or
degraded. Async PDF jobs are covered by the blob storage live smoke.

### Run Without LangSmith Trace

```bash
PAPERINTEL_RUN_ASYNC_JOBS_LIVE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_async_jobs_live_smoke.py
```

### Run With LangSmith Trace

```bash
PAPERINTEL_RUN_ASYNC_JOBS_LIVE=1 \
PAPERINTEL_LIVE_TRACE=1 \
LANGCHAIN_TRACING_V2=true \
LANGSMITH_TRACING=true \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_async_jobs_live_smoke.py
```

### Expected Success Markers

```text
LIVE_AJ_RUN_ID=<run_id>
LIVE_AJ_SESSION_ID=<uuid>
LIVE_AJ_JOB_ID=<uuid>
LIVE_AJ_WORKER_PROCESSED=<job_id>:succeeded
LIVE_AJ_JOB_STATUS=succeeded
LIVE_AJ_WORKSPACE_IDS=1706.03762
LIVE_AJ_INVALID_JOB=<job_id>:failed
LIVE_AJ_CANCELED_JOB=<job_id>:canceled
LIVE_AJ_QDRANT_CLEANUP=success
LIVE_AJ_POSTGRES_CLEANUP=success
1 passed
```

## Common Failures

ROS pytest plugin error:

```text
AttributeError: module 'asyncio' has no attribute 'coroutine'
```

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Missing local PDFs:

```text
Local PDFs are required for CA live smoke
Local PDF is required for PDF live smoke
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
