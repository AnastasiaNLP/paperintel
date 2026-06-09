# Observability Contract

PaperIntel observability has three goals:

- correlate user-visible work across sessions, turns, jobs, papers, agents, and
  provider calls;
- preserve enough durable state to debug quality, failure, fallback, and retry
  behavior;
- keep public/API/MCP output neutral and free of secrets, raw prompts, raw PDF
  text, tracebacks, and internal graph details.

This document defines the public observability contract. It does not introduce
a metrics backend, dashboard, or tracing vendor requirement.

## Current State

| Surface | Current behavior | Gap |
| --- | --- | --- |
| `AgentRun` | Durable per-agent execution record with `session_id`, `job_id`, `agent_name`, `model`, lifecycle status, `termination_reason`, counts, refs, and `details`. | Trace correlation is not standardized. Raw prompt/output policy is documented but not centrally enforced. |
| Agent policies | Agents record `details.policy_applied`, pass `AgentRuntimePolicy.timeout_seconds` to LLM calls, and normalize final timeout outcomes for production-shaped agent runs. | Older agents that do not use the production `AgentRun` policy contract may still need cleanup. |
| LLM provider | `call_text_llm()` logs provider/model and emits `llm.call.completed`, `llm.call.failed`, and `llm.call.timeout` events. | No duration field or trace/correlation payload yet. |
| Workflow worker | Job failures persist `failure_class`, actual `retryable`, and optional `retry_after_seconds`; worker emits started/completed/failed events. | Event coverage is still limited to worker and LLM provider surfaces. |
| Health | Reports Postgres, provider resilience store, Qdrant, LLM/embedding config, and blob store. | No metrics export. |
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
| `duration_ms` | Runtime duration for an operation when measured. | No by default | Yes | Future |
| `trace_id` | External trace/correlation id. | No by default | Yes | Future optional AgentRun details |

## Event Taxonomy

Structured logs and trace spans should use these event names instead of ad hoc
prose. The current logging helper allowlists event fields and drops prompt,
document, traceback, and secret-like fields.

| Event | Required fields |
| --- | --- |
| `session.created` | `session_id` |
| `turn.appended` | `session_id`, `turn_id` |
| `workflow.job.started` | `job_id`, `session_id`, `kind`, `worker_id`, `attempts`, `max_attempts` |
| `workflow.job.completed` | `job_id`, `session_id`, `kind`, `worker_id`, `attempts`, `max_attempts` |
| `workflow.job.failed` | `job_id`, `session_id`, `kind`, `worker_id`, `failure_class`, `retryable`, `attempts`, `max_attempts` |
| `agent.started` | `agent_run_id`, `agent_name`, `session_id`, optional `job_id` |
| `agent.completed` | `agent_run_id`, `agent_name`, `duration_ms`, `termination_reason` |
| `agent.failed` | `agent_run_id`, `agent_name`, `failure_class` or error category |
| `llm.call.started` | `provider`, `model`, `agent_name` or `context_label` |
| `llm.call.completed` | `provider`, `model`, `result_size`, optional `timeout_seconds` |
| `llm.call.failed` | `provider`, `model`, `failure_class` or error category, optional `timeout_seconds` |
| `llm.call.timeout` | `provider`, `model`, `timeout_seconds` |
| `retrieval.search.completed` | `session_id`, optional `paper_id`, result count, duration |
| `provider.failure` | `provider`, operation, `failure_class`, retryable decision |
| `cache.reused_analysis` | `session_id`, `paper_id`, reuse source category |

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

## Timeout And Bounded Execution

`AgentRuntimePolicy.timeout_seconds` is part of the agent policy contract.
Production-shaped agents pass the resolved timeout to `call_text_llm()`, which
uses it as the provider call timeout when supplied.

Timeout output must stay neutral, for example `Answer Agent call timed out`.
It must not include prompts, full user content, provider secrets, or raw provider
exception text. Where an `AgentRun` records a timeout as the final outcome, use
`termination_reason="timeout"` rather than a generic error or fallback reason.

## Future Hardening

Timeout behavior should be standardized in older agents that do not yet use the
production `AgentRun` policy contract.

Structured event coverage should expand beyond worker and LLM provider
surfaces, starting with retrieval search and provider failure events.

Optional trace/correlation fields may be added to `AgentRun.details` without
exposing them in default REST/MCP payloads.

Live smoke runbooks should list expected observability markers and how to
disable external tracing locally.

Metrics export should be evaluated only after structured events are stable. Do
not commit to Prometheus, Grafana, or a hosted tracing system before that
contract is implemented.
