# Async Jobs

PaperIntel has a Postgres-backed workflow job queue for long-running analysis
work. The queue is intentionally small and explicit: API/MCP calls enqueue jobs,
a separate worker process claims jobs, and clients poll job status.

This is the current async foundation. It is not a full scheduler, retry system,
or deployment supervisor.

## Supported Job Kinds

The worker currently supports:

- `analyze_paper`: analyze one paper URL.
- `analyze_selected`: analyze papers selected from the current discovery
  shortlist.

Other job kinds may exist in the type model for future work, but the default
worker only claims supported kinds. Unsupported jobs are not picked up by the
normal worker command.

## Run The Worker

Start a long-running worker:

```bash
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m workers
```

Run one job and exit:

```bash
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m workers --once
```

Limit the worker to one kind:

```bash
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m workers --kind analyze_paper
```

Use a stable worker ID when running more than one worker process:

```bash
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m workers --worker-id worker-a
```

The worker must run separately from the REST API or MCP server. Starting the API
creates jobs; it does not process them.

## Lifecycle Contract

Workflow jobs use this lifecycle:

```text
queued -> running -> succeeded
queued -> running -> failed
queued -> canceled
running -> canceled
```

Important invariants:

- Workers start jobs only through `claim_next(worker_id=...)`.
- `claim_next` changes `queued` to `running` and increments `attempts`.
- Workers must not call `mark_running` after `claim_next`.
- `mark_succeeded` and `mark_failed` are only valid from `running`.
- `mark_canceled` is valid from `queued` or `running`.
- Terminal transitions are guarded by the repository.

A successful worker run writes `result_json`. A failed worker run writes
`error_json`.

## REST Examples

Create a session:

```bash
SESSION_ID=$(
  curl -s -X POST http://127.0.0.1:8000/sessions \
    -H 'content-type: application/json' \
    -d '{"persona":"engineer"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])"
)
```

Queue one paper analysis job:

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/jobs/analyze-paper" \
  -H 'content-type: application/json' \
  -d '{"paper_url":"https://arxiv.org/abs/1706.03762"}'
```

Queue analysis for selected discovery candidates:

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/jobs/analyze-selected"
```

List jobs for a session:

```bash
curl -s "http://127.0.0.1:8000/sessions/$SESSION_ID/jobs"
```

Get one job:

```bash
curl -s "http://127.0.0.1:8000/jobs/$JOB_ID"
```

Cancel a queued or running job:

```bash
curl -s -X POST "http://127.0.0.1:8000/jobs/$JOB_ID/cancel"
```

HTTP status notes:

- Enqueue endpoints return `202` with a `WorkflowJob` body.
- Missing sessions return `404`.
- Invalid job input returns `400` or request validation `422`, depending on
  where validation fails.
- Invalid lifecycle transitions return `409`.

## MCP Tools

The MCP server exposes these job tools:

- `enqueue_analyze_paper(session_id, paper_url)`
- `enqueue_analyze_selected(session_id)`
- `get_workflow_job(job_id)`
- `list_workflow_jobs(session_id, limit)`
- `cancel_workflow_job(job_id)`

Example prompt for an MCP client:

```text
Create a PaperIntel session and queue analysis for https://arxiv.org/abs/1706.03762.
Then show me the workflow job status.
```

The MCP client should call `create_session`, then `enqueue_analyze_paper`, then
`get_workflow_job` or `list_workflow_jobs`. The worker still needs to be running
separately for queued jobs to complete.

## Live Smoke

The async jobs live smoke verifies REST enqueue, worker execution, REST status,
MCP status, deterministic failure, and cancel behavior on the real stack.

Run it with:

```bash
PAPERINTEL_RUN_ASYNC_JOBS_LIVE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m dotenv run -- \
.venv/bin/python -m pytest -s tests/live/test_async_jobs_live_smoke.py
```

The runbook in `tests/live/README.md` lists expected success markers and tracing
options.

## Current Limits

- No automatic retries or retry backoff yet.
- No progress percentages yet.
- No job idempotency keys yet.
- No job budget or quota enforcement yet.
- No process supervisor, systemd unit, or Docker worker service yet.
- Canceling a running job marks the job record canceled; it does not interrupt
  an already running LLM, embedding, HTTP, or vector-store call.
- No async comparison or synthesis jobs yet.
- Local PDF analysis is synchronous through REST multipart upload, MCP local
  path analysis, or direct service calls. The current async worker supports URL
  analysis jobs; async PDF upload jobs require shared PDF/object storage.
- Job results are stored in Postgres as JSON transport snapshots, not as a
  separate event stream.
