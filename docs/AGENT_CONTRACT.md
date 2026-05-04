# Agent Contract

This document defines the production contract for controlled agents in PaperIntel.
Use it when adding or changing nodes that make real decisions or produce
publishable output.

## When To Use AgentRun

Wrap a node in `AgentRun` when it does at least one of these:

- makes a real decision the deterministic pipeline cannot safely hardcode
- produces publishable user-facing output
- reviews, downgrades, repairs, or rejects another agent's output
- needs auditability for cost, policy, fallback, or failure analysis

Do not wrap pure lifecycle nodes only because they live in the graph. Finalizers,
routers, persistence helpers, and fixed API calls are processors or workflow nodes,
not agents.

## Required Fields

Every run must set:

- `agent_name`: stable literal name, for example `report` or `evidence_critic`
- `session_id`: from `RunnableConfig.configurable.session_id` when available
- `job_id`: from `RunnableConfig.configurable.job_id` when available
- `model`: LLM model name, or `None` for deterministic agents
- `input_refs`: state locations and previous run ids used as inputs
- `output_ref`: state or artifact location for the produced output
- `iteration_count`: agent loop iterations, not raw LLM calls
- `llm_call_count`: raw LLM calls, including repairs
- `details.policy_applied`: resolved `AgentRuntimePolicy` snapshot

For inter-agent traceability, include both state location and upstream run ids
when available:

```python
input_refs = ["state:report", report_run.id]
```

## Lifecycle

Every agent follows the same shape:

```python
policy = resolve_agent_policy(agent_name, config, strict=True)
run = AgentRun(
    agent_name=agent_name,
    session_id=session_id,
    job_id=job_id,
    input_refs=input_refs,
    model=model,
    iteration_count=1,
)
run.details["policy_applied"] = policy.model_dump(mode="json")

try:
    # do work
    run.complete(output_ref="state:...", details={...})
except Exception:
    run.fail(output_ref="state:errors", details={...})

persistence.save(run)
return {**result, "agent_runs": [run]}
```

Never return a pending run. Finalize with `complete`, `fail`, or `fallback`
before saving or returning.

## Termination Reasons

- `success`: agent completed normally
- `error`: agent failed and the graph should use existing error handling
- `skipped`: agent was not needed for this state, for example no report to review
- `fallback`: agent attempted the task and used fallback behavior
- `max_iter`: bounded loop stopped by iteration limit
- `timeout`: runtime timeout stopped the agent
- `budget`: session or run budget stopped the agent

Use `skipped` instead of `fallback` when the agent did not need to run.
Use `fallback` only when the agent tried and then degraded to an alternate path.

## Runtime Policy

Policies are resolved through:

```python
resolve_agent_policy(agent_name, config, strict=True)
```

Production code uses `strict=True`. Unknown agent names must fail loudly so new
agents cannot run without an explicit policy.

Per-call overrides use full replacement through `RunnableConfig`:

```python
config = {
    "configurable": {
        "agent_policy_overrides": {
            "report": AgentRuntimePolicy(...)
        }
    }
}
```

No partial merge is supported. If a caller wants to change one field, it should
construct a full replacement policy.

## Persistence

Agents save through the persistence seam:

```python
persistence.save(run)
```

Default behavior is `NoopAgentRunPersistence`. Tests and future repositories can
inject another persistence implementation through `RunnableConfig`:

```python
config["configurable"]["agent_run_persistence"] = persistence
```

Save both successful and failed runs. Failed runs are part of the audit trail.

## Current Agents

Current production-shaped agents:

- `report`: LLM-heavy publishable report producer
- `evidence_critic`: deterministic critic that can pass through, downgrade, or skip

Current runtime invariant:

- `report` enforces `llm_call_count <= policy.max_tool_calls` post-hoc and records
  `policy_warning`, `policy_max_tool_calls`, and `actual_llm_call_count` when the
  limit is exceeded.
- `evidence_critic` records `skip_review_on_no_report` as documented skipped
  behavior, not as fallback noise.
