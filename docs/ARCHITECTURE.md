# PaperIntel Architecture

PaperIntel is a research intelligence system for AI/ML papers. It can analyze
known papers by URL, answer grounded questions about analyzed papers, and
discover recent candidate papers for a research topic.

## Implemented System

```text
┌──────────────────────────────────────────────────────────────────┐
│                         TRANSPORT LAYER                          │
│                                                                  │
│  FastAPI REST adapter                 MCP stdio server           │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     APPLICATION SERVICE                          │
│                                                                  │
│  PaperIntelService                                               │
│  - create_session                                                │
│  - analyze_paper                                                 │
│  - ask_question                                                  │
│  - compare_papers / synthesize_papers                           │
│  - discover_papers / select_papers                               │
│  - analyze_selected_papers                                       │
│  - list/get paper workspaces and latest comparison               │
│  - enqueue/get/list/cancel workflow jobs                         │
│  - get_session / list_turns                                      │
│  - health                                                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       CHAT HANDLER                               │
│                                                                  │
│  - writes user turns before graph invocation                     │
│  - writes assistant turns after graph invocation                  │
│  - routes paper URLs to the analysis graph                       │
│  - routes discovery requests to the discovery graph               │
│  - routes selection turns while session.phase == selection        │
│  - invokes batch analysis for selected discovery candidates       │
│  - persists completed analysis artifacts after graph invocation   │
│  - routes questions to the conversation graph                    │
│  - passes session_store, retrieval_layer, and                    │
│    agent_run_persistence through RunnableConfig                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
          ┌────────────────────┬────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ANALYSIS GRAPH  │  │ CONVERSATION     │  │ DISCOVERY GRAPH  │
│                  │  │ GRAPH            │  │                  │
│ supervisor       │  │ intent_router    │  │ research         │
│   ↓              │  │   ├─ qa_*        │  │ strategist       │
│ ingestion        │  │   │  ↓           │  │   ↓              │
│   ↓              │  │ retrieval        │  │ deterministic    │
│ extraction       │  │ planner          │  │ searcher         │
│   ↓              │  │   ↓              │  │   ↓              │
│ benchmark        │  │ answer_agent     │  │ selection        │
│   ↓              │  │   ↓              │  │ advisor          │
│ readiness        │  │ citation_critic  │  │   ↓              │
│   ↓              │  │   ├─ repair      │  │ selection phase  │
│ report           │  │   └─ END         │  │                  │
│   ↓              │  │   ├─ clarify     │  │                  │
│ evidence_critic  │  │   ├─ analyze     │  │                  │
│   ↓              │  │   └─ discover    │  │                  │
│ report_finalize  │  │                  │  │                  │
│   ↓              │  │                  │  │                  │
│ chunk_and_index  │  │                  │  │                  │
│   ├─ next paper  │  │                  │  │                  │
│   ├─ compare     │  │                  │  │                  │
│   └─ END         │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Analysis Flow

The analysis graph handles explicit arXiv URLs, PDFs, and URL batches:

1. `supervisor` validates routing state.
2. `ingestion` fetches arXiv metadata, Semantic Scholar metadata, and PDF text.
3. `extraction` extracts method, novelty, components, and limitations.
4. `benchmark` extracts tasks, datasets, metrics, and result context.
5. `readiness` checks implementation maturity and external resources.
6. `report` creates an engineer-facing report and records an `AgentRun`.
7. `evidence_critic` reviews the report and can downgrade unsupported claims.
8. `report_finalize` stores the analyzed paper into session state.
9. `chunk_and_index` chunks the paper, stores chunks in Postgres, embeds them,
   upserts vectors into Qdrant, and marks the paper active only after successful
   indexing.
10. Batch runs loop to the next paper or compare multiple completed papers.

Indexing failures are non-fatal: analysis can complete even if retrieval setup is
unavailable. In that case the paper is not added to `active_paper_ids` and QA
will not treat it as retrievable.

When a batch contains multiple successfully completed papers, the analysis graph
runs the existing batch comparator and may return `comparison_markdown` plus a
structured comparison report. This is a post-analysis artifact built from the
structured outputs of the analyzed papers.

After analysis graph invocation, `ChatHandler` persists durable artifacts in the
service/handler layer. Graph nodes do not import database repositories. The
write path reads completed `PaperSlot` state and stores per-paper workspaces;
when a batch comparison exists, it stores a session-scoped comparison artifact.
Failed analysis results are not written as successful artifacts.

## Conversation QA Flow

The conversation graph handles questions about papers that were successfully
indexed in the current session:

1. `intent_router` classifies the user message and resolves referenced papers.
2. `retrieval_planner` builds a persona-aware retrieval plan with chunk type
   priorities and section hints.
3. `PostgresQdrantRetrievalLayer` retrieves chunks and assembles an
   `EvidenceBundle`.
4. `answer_agent` writes a persona-aware answer grounded in retrieved evidence.
5. `citation_critic` checks the answer against evidence and can trigger bounded
   repair.

Repair is bounded by `MAX_REPAIR_ITERATIONS = 2` and centralized in
`services/repair.py`.

`PaperIntelService.compare_papers` is the request-driven comparison path. It
loads durable `PaperWorkspace` artifacts, runs `comparison_analyst`, and
persists a new `ComparisonArtifact` with `producer="comparison_analyst"`.
Each successful request creates a new artifact; `GET /comparison` only reloads
the latest artifact and does not run an LLM.

`PaperIntelService.synthesize_papers` runs the dedicated `synthesis_agent` over
durable workspaces. It returns persona-aware synthesis with citations and may
use the latest relevant comparison as optional context. It does not persist a
`SynthesisArtifact`.

PaperIntel therefore has three multi-paper paths:

- Batch comparison: produced automatically when multiple papers are analyzed
  together. It writes `ComparisonArtifact` with `producer="batch_comparator"`.
- Request-driven comparison: produced on demand by `comparison_analyst` from
  durable workspaces. It writes `ComparisonArtifact` with
  `producer="comparison_analyst"`.
- Request-driven synthesis: produced on demand by `synthesis_agent`; it returns
  a response but does not persist a synthesis artifact.

`agents/comparator.py` remains as the legacy batch-analysis comparator. It
coexists with `comparison_analyst`; both write the same comparison artifact type
so callers have one persisted comparison surface.

## Evaluation Layer

The evaluation layer is built on persisted artifacts rather than transient
graph state. It has three complementary paths:

- Deterministic paper evaluation: validates golden JSONL labels, exports
  `PaperWorkspace` rows, and checks benchmark rows, readiness fields, and
  keyword coverage. This path is stable enough for CI-style gating.
- Deterministic CA structural checks: validate comparison and synthesis output
  shape, citation coverage, selected paper coverage, producer markers, and
  AgentRun policy metadata. These are CI-safe structural checks, not quality
  claims.
- Judge evaluation: loads versioned rubrics from `evaluation/rubrics/` and can
  score report, comparison, and synthesis fields through the configured LLM
  provider. This path is a manual or scheduled quality gauge, not a normal CI
  gate, because judge scores are non-deterministic.

The repository includes a schema-clean 30-paper golden dataset for project-level
evaluation, published on Hugging Face as
[AIAnastasia/arxiv-papers](https://huggingface.co/datasets/AIAnastasia/arxiv-papers).
The measured v0.1 baseline identifies benchmark extraction on complex PDF
tables as the main known quality weakness.

## Discovery Flow

The discovery graph handles topic-level requests such as "find recent papers
about retrieval augmented generation":

1. `research_strategist` converts the topic into 2-3 short arXiv queries.
2. `ArxivSearchProvider` calls the arXiv API with retry/backoff and rate-limit
   spacing.
3. `Searcher` deterministically deduplicates, scores, ranks, and persists
   `SearchCandidate` rows.
4. `selection_advisor` writes a shortlist and asks the user to choose by display
   number.
5. `ChatHandler` sets `session.phase = selection`; the next user selection is
   parsed deterministically and stored as selected candidate IDs.
6. `PaperIntelService.analyze_selected_papers` resolves selected candidate IDs
   to URLs, invokes the existing analysis graph in batch mode, and marks
   candidates `analyzed` only after successful analysis.
7. Successfully analyzed selected papers are indexed and become available for
   retrieval-backed QA, request-driven comparison, and dedicated synthesis.

Only `research_strategist` and `selection_advisor` are LLM agents. Search,
ranking, and selection parsing are deterministic components.

## Data Layer

Postgres stores durable product state:

- `sessions`
- `turns`
- `agent_runs`
- `structured_errors`
- `paper_chunks`
- `search_candidates`
- `paper_workspaces`
- `comparison_artifacts`
- `workflow_jobs`

Qdrant stores chunk vectors. Point IDs are deterministic UUID5 values derived
from stable chunk IDs, so repeated indexing updates instead of duplicating.

## Artifact Persistence

Artifact persistence is intentionally narrow and Postgres-backed:

- `paper_workspaces` stores one session-scoped snapshot per analyzed paper,
  keyed by `(session_id, paper_id)`.
- A workspace contains finalized report JSON, method extraction JSON,
  benchmark JSON, readiness JSON, and the markdown report.
- `comparison_artifacts` stores session-scoped comparison artifacts for groups
  of papers, including `paper_ids`, `comparison_markdown`, and a report JSON
  producer marker (`batch_comparator` or `comparison_analyst`).
- REST and MCP read paths can reload these artifacts without re-running
  analysis.
- Re-analysis of the same paper in the same session uses last-write-wins
  upsert semantics.

The workflow job layer is intentionally narrow and Postgres-backed:

- `workflow_jobs` stores queued/running/terminal job records with JSON input,
  result, and error payloads.
- Workers claim jobs through repository lifecycle methods and run supported job
  kinds outside the request/response path.
- REST and MCP expose enqueue, status, list, and cancel surfaces.
- The current worker supports URL analysis jobs and selected-paper analysis
  jobs.

This layer does not include S3/object storage, paper cache versioning,
retry/backoff scheduling, process supervision, job budgets, async comparison or
synthesis jobs, or PDF/page-image asset storage. Those are separate later
hardening layers.

## AgentRun Contract

Production-shaped agents record:

- agent name
- session ID
- input references
- output reference
- model
- LLM call count
- termination reason
- status
- policy snapshot

This contract is implemented for `report`, `evidence_critic`, and the QA team:
`intent_router`, `retrieval_planner`, `answer_agent`, and `citation_critic`.
It is also implemented for the discovery agents: `research_strategist` and
`selection_advisor`. Other analysis processors are intentionally still simpler
pipeline processors.

See [AGENT_CONTRACT.md](AGENT_CONTRACT.md) for implementation details.

## Current Limitations

- Synchronous analysis and discovery endpoints remain available. Async analysis
  is available through workflow job enqueue/status surfaces and a separately
  running worker.
- Discovery currently searches arXiv only.
- Artifact persistence is session-scoped. Global paper reuse without re-analysis
  is deferred to a future PaperCache layer.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication, rate limiting, and deployment hardening are future work.
