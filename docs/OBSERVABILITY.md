# Observability Contract

PaperIntel observability has three goals:

- correlate user-visible work across sessions, turns, jobs, papers, agents, and
  provider calls;
- preserve enough durable state to debug quality, failure, fallback, and retry
  behavior;
- keep public/API/MCP output neutral and free of secrets, raw prompts, raw PDF
  text, tracebacks, and internal graph details.

This document defines the public observability contract. Metrics are exported
in Prometheus text format, but the contract does not require a specific
Prometheus/Grafana deployment or hosted tracing vendor.

## Current State

| Surface | Current behavior | Gap |
| --- | --- | --- |
| `AgentRun` | Durable per-agent execution record with `session_id`, `job_id`, `agent_name`, `model`, lifecycle status, `termination_reason`, counts, refs, and `details`; persistence links optional trace metadata into `details.observability` and emits lifecycle events. | Raw prompt/output policy is documented but not centrally enforced for every ad hoc details payload. |
| Agent policies | Agents record `details.policy_applied`, production LLM paths pass `AgentRuntimePolicy.timeout_seconds` or registered pipeline policy timeouts to `call_text_llm()`, and `AgentRun` paths use the shared LLM timeout classifier for final outcomes. | External trace correlation depends on runtime trace context being available. |
| LLM provider | `call_text_llm()` logs provider/model and emits `llm.call.completed`, `llm.call.failed`, and `llm.call.timeout` events with `duration_ms`. Safe events automatically include `trace_id` when available. | No vendor-specific span export is implemented by PaperIntel itself. |
| Workflow worker | Job failures persist `failure_class`, actual `retryable`, and optional `retry_after_seconds`; worker emits started events and terminal completed/failed events with `duration_ms`. | No UI-level progress percentage yet. |
| Retrieval | Retrieval search emits `retrieval.search.completed` with correlation ids, result count, `duration_ms`, and optional trace correlation, without query text or chunk text. | Query planning details are not exported as metrics. |
| Provider failures | arXiv, Semantic Scholar, OpenAI embeddings, S3/MinIO blob storage, Qdrant, and Postgres-backed resilience-store degradation emit `provider.failure` with neutral failure class and retry decision. | LLM provider breaker/limiter coverage remains future resilience work. |
| Health | Reports Postgres, provider resilience store, Qdrant, LLM/embedding config, and blob store. | Health is status-oriented; metrics are exported separately. |
| Metrics | `/metrics` returns safe in-process Prometheus text metrics derived from structured events: `paperintel_events_total` and `paperintel_event_duration_ms`. | Metrics reset on process restart and are not a durable event stream. |
| REST/MCP | Public result payloads intentionally omit raw `AgentRun` internals and preserve neutral failure classes/metadata where needed. | Correlation IDs are not consistently surfaced for every async/sync operation. |
| Live smoke tests | Print stable `LIVE_*` markers for run/session/job/resource IDs and cleanup. | Marker taxonomy is test-specific, not a general event contract. |

## Public Boundary

Public REST/MCP responses may include:

- `session_id`
- `job_id`
- stable status values
- `failure_class`
- neutral user-facing messages
- `retryable` when it means the job will actually retry
- `retry_after_seconds` when it is part of job retry status
- reusable-analysis metadata such as `analysis_reused`

Public REST/MCP responses must not include:

- raw prompts or full model inputs
- full PDF text or extracted long document text
- provider API keys, object-store credentials, database URLs, or signed URLs
- Python tracebacks
- raw provider response bodies when they may contain request details
- internal workflow implementation wording or circuit-breaker implementation wording

Operational logs and persisted diagnostics may include technical details when
they are needed for debugging, but they must still avoid secrets and raw
document/model payloads.

## Correlation Fields

These fields are the canonical vocabulary for future structured logs, traces,
metrics, and runbook output.

| Field | Meaning | Public response | Logs | Persisted |
| --- | --- | --- | --- | --- |
| `session_id` | User research session. | Yes | Yes | Sessions, turns, workspaces, jobs, agent runs |
| `turn_id` | Conversation turn. | Yes when relevant | Yes | Turns |
| `job_id` | Async workflow job. | Yes | Yes | Workflow jobs, agent runs |
| `agent_run_id` | Durable agent execution id. | No by default | Yes | Agent runs, structured errors when relevant |
| `agent_name` | Stable agent literal. | No by default | Yes | Agent runs |
| `paper_id` | Canonical paper/workspace id. | Yes | Yes | Workspaces, chunks, jobs, references |
| `provider` | External provider or dependency, for example `anthropic`, `openai`, `arxiv`, `qdrant`. | No by default | Yes | Diagnostics where relevant |
| `model` | LLM or embedding model name. | No by default | Yes | Agent runs and chunk metadata |
| `failure_class` | Provider-neutral failure class. | Yes for failures/jobs | Yes | Jobs and structured errors |
| `retryable` | Whether the current job will actually retry. | Yes for jobs | Yes | Workflow job error JSON |
| `retry_after_seconds` | Provider/shared-breaker delay hint. | Yes for jobs | Yes | Workflow job error JSON |
| `operation` | Provider or dependency operation name. | No by default | Yes | Diagnostics where relevant |
| `kind` | Async workflow job kind. | No by default | Yes | Workflow jobs |
| `worker_id` | Worker process identifier for async job execution. | No | Yes | Workflow job diagnostics |
| `attempts` | Current async job attempt count. | Yes for jobs | Yes | Workflow job error JSON and diagnostics |
| `max_attempts` | Async job retry budget. | No by default | Yes | Workflow job diagnostics |
| `timeout_seconds` | Configured provider-call timeout. | No by default | Yes | Provider-call diagnostics |
| `status` | Stable lifecycle status. | Yes when relevant | Yes | Sessions, jobs, agent runs |
| `result_size` | Size of a generated or returned result, usually character count. | No by default | Yes | Provider-call diagnostics |
| `result_count` | Count of returned results. | No by default | Yes | Retrieval diagnostics |
| `duration_ms` | Runtime duration for a provider call, retrieval search, workflow job, or agent run when measured. | No by default | Yes | Provider-call, retrieval, workflow-job, and AgentRun diagnostics |
| `trace_id` | External trace/correlation id resolved from explicit env correlation ids or active LangSmith run context. | No by default | Yes | Optional `AgentRun.details.observability.trace_id` |

## Event Taxonomy

Structured logs and trace spans should use these event names instead of ad hoc
prose. The current logging helper allowlists event fields and drops prompt,
document, traceback, and secret-like fields.

| Event | Required fields |
| --- | --- |
| `session.created` | `session_id` |
| `turn.appended` | `session_id`, `turn_id` |
| `workflow.job.started` | `job_id`, `session_id`, `kind`, `worker_id`, `attempts`, `max_attempts` |
| `workflow.job.completed` | `job_id`, `session_id`, `kind`, `worker_id`, `attempts`, `max_attempts`, `duration_ms` |
| `workflow.job.failed` | `job_id`, `session_id`, `kind`, `worker_id`, `failure_class`, `retryable`, `attempts`, `max_attempts`, `duration_ms` |
| `agent.started` | `agent_run_id`, `agent_name`, optional `session_id`, optional `job_id`, optional `model` |
| `agent.completed` | `agent_run_id`, `agent_name`, `status`, `termination_reason`, `duration_ms`, optional `session_id`, optional `job_id`, optional `model` |
| `agent.failed` | `agent_run_id`, `agent_name`, `status`, `termination_reason`, `failure_class`, `duration_ms`, optional `session_id`, optional `job_id`, optional `model` |
| `llm.call.started` | `provider`, `model`, `agent_name` or `context_label` |
| `llm.call.completed` | `provider`, `model`, `result_size`, `duration_ms`, optional `timeout_seconds` |
| `llm.call.failed` | `provider`, `model`, `duration_ms`, `failure_class` or error category, optional `timeout_seconds` |
| `llm.call.timeout` | `provider`, `model`, `timeout_seconds`, `duration_ms` |
| `retrieval.search.completed` | `session_id`, optional `paper_id`, `result_count`, `duration_ms` |
| `provider.failure` | `provider`, `operation`, `failure_class`, `retryable`, optional `retry_after_seconds` |
| `cache.reused_analysis` | `session_id`, `paper_id`, reuse source category |

`provider.failure` uses provider labels such as `arxiv`, `semantic_scholar`,
`openai`, `s3`, `qdrant`, and `postgres`. Operations must be stable literals
such as `embeddings`, `search`, `head_object`, `upsert`, or
`provider_rate_limiter.reserve_slot`; they must not include object keys, URLs,
queries, prompts, or raw exception text.

## Metrics Export

`GET /metrics` exposes process-local Prometheus text metrics:

- `paperintel_events_total{event=..., ...}` counts safe structured events.
- `paperintel_event_duration_ms_bucket/count/sum{event=..., ...}` records
  event durations when a safe event includes `duration_ms`.

Metric labels are intentionally low cardinality:

- always: `event`
- optional safe labels: `agent_name`, `failure_class`, `kind`, `model`,
  `operation`, `provider`, `status`, `termination_reason`

High-cardinality correlation fields such as `session_id`, `job_id`,
`agent_run_id`, `paper_id`, and `trace_id` are allowed in logs but excluded from
metric labels.

## AgentRun Contract

`AgentRun` is the durable execution record for controlled agents. It should be
created only for nodes that make real decisions, produce publishable output, or
review/repair another agent output. Pure lifecycle functions remain workflow
nodes, not agents.

Every production-shaped agent run should keep:

- `agent_name`
- `session_id` when available
- `job_id` when available
- `model` for LLM agents
- `input_refs`
- `output_ref`
- `iteration_count`
- `llm_call_count`
- final `status`
- `termination_reason`
- `details.policy_applied`

AgentRun details may add a future neutral observability block, for example:

```json
{
  "observability": {
    "trace_id": "optional-external-trace-id",
    "duration_ms": 1234
  }
}
```

Do not store raw prompts, full model outputs, full PDF text, secrets, or signed
object URLs in `AgentRun.details`.

Agent started events are emitted when production-shaped agents create their
`AgentRun`. Terminal lifecycle events are emitted from the persistence boundary
and deduplicated per persistence instance. These events include correlation
identifiers and lifecycle metadata, but they do not include `input_refs`,
`output_ref`, prompts, evidence text, model output, or document content.

## Timeout And Bounded Execution

`AgentRuntimePolicy.timeout_seconds` is part of the agent policy contract.
Production-shaped agents pass the resolved timeout to `call_text_llm()`, which
uses it as the provider call timeout when supplied. Pipeline LLM nodes that do
not create `AgentRun` records use registered default policies so provider calls
are still bounded.

Timeout output must stay neutral, for example `Answer Agent call timed out`.
It must not include prompts, full user content, provider secrets, or raw provider
exception text. Where an `AgentRun` records a timeout as the final outcome, use
`termination_reason="timeout"` rather than a generic error or fallback reason.

## Future Hardening

Structured event coverage should expand to the remaining operational surfaces.

Trace/correlation fields are stored in `AgentRun.details.observability` when
available:

```json
{
  "observability": {
    "trace_id": "external-trace-id",
    "duration_ms": 1234
  }
}
```

Trace IDs are resolved from explicit correlation environment variables first
(`PAPERINTEL_TRACE_ID`, `LANGSMITH_TRACE_ID`, `LANGCHAIN_TRACE_ID`,
`LANGCHAIN_RUN_ID`) and then, when available, from the active LangSmith
`get_current_run_tree()` context. They are not exposed in default REST/MCP
payloads and are not used as Prometheus labels.

Live smoke runbooks should list expected observability markers and how to
disable external tracing locally.

Future metrics hardening may add deployment-level scraping configuration,
Grafana dashboards, and process labels. Do not add user/session/job identifiers
as metric labels.
