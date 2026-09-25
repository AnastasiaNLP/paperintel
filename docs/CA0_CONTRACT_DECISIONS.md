# CA.0 Contract Decisions

This note fixes the implementation contract for the Comparison Analyst and
Synthesis Agent phase. It is intentionally narrow: CA.0 does not implement the
new agents. It records the source-of-truth decisions that CA.1 and CA.2 must
follow.

## Current State

The project already has a batch comparator in `agents/comparator.py`. It runs at
the end of multi-paper analysis, builds a deterministic benchmark matrix, asks an
LLM for trade-off reasoning, and returns a `ComparisonReport` plus markdown.

The durable artifact layer is now available:

- `PaperWorkspace` stores per-paper structured artifacts.
- `ComparisonArtifact` stores session-scoped comparison output.
- `comparison_artifacts` is the single Postgres table for persisted comparisons.

The existing REST and MCP surfaces expose latest comparison retrieval and a
`synthesize_papers` entry point, but synthesis currently routes through normal
conversation QA rather than a dedicated synthesis agent.

## Decisions

### 1. Keep the existing comparator as the batch path

The existing comparator is not deleted in CA.0. It remains the automatic
post-analysis comparison path for batch analysis.

Reasoning:

- removing it now would regress the discovery -> select -> analyze-selected
  batch flow;
- it already provides useful deterministic matrix construction;
- batch-triggered comparison and conversation-triggered comparison are different
  product actions.

### 2. Add Comparison Analyst as a separate request-driven agent

Comparison Analyst is a new production-shaped LLM agent. It is separate from the
existing batch comparator.

Input:

- two or more persisted `PaperWorkspace` records from the current session;
- optional user comparison prompt or constraints;
- no transient graph-only paper state as the source of truth.

Output:

- structured `ComparisonReport`;
- comparison markdown;
- persisted `ComparisonArtifact`.

The agent uses the AgentRun lifecycle and runtime policy contract from
`docs/AGENT_CONTRACT.md`.

### 3. Use one comparison artifact type and one table

Both comparison paths persist to the same `comparison_artifacts` table and use
the same `ComparisonArtifact` model.

There will not be separate artifact tables for batch comparison and analyst
comparison.

Producer tracking should be stored in the structured report JSON, not as a new
database column in CA.0:

- `producer = "batch_comparator"` for the existing batch path;
- `producer = "comparison_analyst"` for the new request-driven agent.

This keeps the database schema stable while still making artifact provenance
auditable. CA.1 should add the producer field to the report schema or report
metadata in a backwards-compatible way.

### 4. Synthesis Agent uses existing personas only

Synthesis Agent must use the current session persona contract:

- `engineer`
- `researcher`
- `techlead`

CA phase does not introduce `founder` or any new persona. Adding personas would
require a separate schema and prompt migration across conversation, discovery,
retrieval planning, answer generation, and tests.

### 5. Synthesis Agent is not a normal QA wrapper

The current `synthesize_papers` service method routes a generic synthesis prompt
through the conversation graph. CA.2 replaces or extends that behavior with a
dedicated production-shaped Synthesis Agent.

Input:

- active persisted `PaperWorkspace` records;
- session persona;
- optional user synthesis prompt;
- optionally latest `ComparisonArtifact` as supporting context.

Output:

- persona-aware narrative answer;
- citations to the source paper ids;
- recommended next steps or interpretation appropriate to the persona.

Synthesis output is not persisted as `ComparisonArtifact`. If persistence is
needed, CA.2 should introduce a separate synthesis artifact contract rather than
overloading comparison artifacts.

### 6. Deterministic eval remains structural for CA outputs

Comparison and synthesis contain judgment-heavy free text. Deterministic eval
cannot validate their semantic quality.

Deterministic checks for CA.4 are limited to:

- output schema validation;
- artifact persistence;
- all requested papers represented in output;
- citations/reference ids point to papers in the active session;
- no dropped paper ids in comparison/synthesis payloads.

Quality checks require G-Eval rubrics and remain gauge metrics, not CI gates.
Rubrics should cover:

- comparison balance across papers;
- trade-off grounding in durable artifacts;
- synthesis justification for the selected persona;
- citation faithfulness.

## CA Phase Sequence

1. CA.1 Comparison Analyst
   - Add runtime policy for `comparison_analyst`.
   - Build from durable `PaperWorkspace` inputs.
   - Persist `ComparisonArtifact` with producer metadata.
   - Add service hook and structural tests.

2. CA.2 Synthesis Agent
   - Add runtime policy for `synthesis_agent`.
   - Use existing personas only.
   - Build from durable workspaces and optional latest comparison.
   - Add service hook and structural tests.

3. CA.3 REST and MCP hooks
   - Add explicit compare request endpoint/tool.
   - Wire synthesis endpoint/tool to the new synthesis agent.

4. CA.4 Eval coverage
   - Deterministic structural tests in CI.
   - G-Eval rubrics as non-gating quality gauges.

5. CA.5 Live smoke
   - discover -> select two or more papers -> analyze-selected;
   - run request-driven comparison explicitly;
   - run synthesis explicitly;
   - assert both new agents record AgentRuns and durable outputs.

## Non-Goals

- No persona enum expansion in this phase.
- No new comparison artifact table.
- No async job queue work; sync MVP first.
- No attempt to make deterministic eval judge free-text quality.
