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
│  - synthesize_papers                                             │
│  - discover_papers / select_papers                               │
│  - analyze_selected_papers                                       │
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

`PaperIntelService.synthesize_papers` is a product-facing wrapper over this same
conversation QA flow. It uses a default synthesis prompt, or a caller-provided
prompt, to compare and synthesize active papers through retrieval-backed QA with
citations. It does not replace the batch comparator; it is an on-demand
conversational synthesis path over indexed paper chunks.

PaperIntel therefore has two comparison paths:

- Batch comparison: produced automatically when multiple papers are analyzed
  together. It compares structured analysis outputs and returns a comparison
  artifact.
- Conversational synthesis: produced on demand through QA over active paper
  chunks, with citations.

This closes the current discovery plus comparison/synthesis MVP. Dedicated
`comparison_analyst` and `synthesis_agent` components are intentionally
deferred until artifact persistence exists. Without durable finalized reports,
method extraction outputs, benchmark results, readiness outputs, and comparison
reports, those agents would have to rely on transient graph state, markdown
scraping, or re-analysis, which would make the design brittle.

`agents/comparator.py` is a known transitional component. It remains the batch
analysis comparator used after multi-paper analysis, and future work should
migrate that behavior into a `comparison_analyst` path once durable artifacts
can be loaded directly.

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
   retrieval-backed QA and synthesis through the conversation graph.

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

Qdrant stores chunk vectors. Point IDs are deterministic UUID5 values derived
from stable chunk IDs, so repeated indexing updates instead of duplicating.

## Next Artifact Persistence Slice

The next persistence slice is intentionally narrow:

- Postgres tables for finalized reports.
- Postgres tables for method extraction outputs.
- Postgres tables for benchmark results.
- Postgres tables for readiness results.
- Postgres tables for comparison reports.
- Repository methods to reload these artifacts without re-running analysis.

This slice does not include S3/object storage, paper cache versioning,
outbox/job processing, or PDF/page-image asset storage. Those are separate
later hardening layers.

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

- Analysis and discovery are synchronous through REST and MCP.
- Discovery currently searches arXiv only.
- Artifact persistence for finalized reports, extraction, benchmarks,
  readiness, and comparison reports is not implemented yet.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication, rate limiting, and deployment hardening are future work.
