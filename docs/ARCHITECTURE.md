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
│  - discover_papers / select_papers                               │
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
│  - routes questions to the conversation graph                    │
│  - passes session_store, retrieval_layer, and                    │
│    agent_run_persistence through RunnableConfig                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
          ┌────────────────────┬────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│        ANALYSIS GRAPH         │       │      CONVERSATION GRAPH      │
│                              │       │                              │
│ supervisor                   │       │ intent_router                │
│   ↓                          │       │   ├─ qa_*                    │
│ ingestion                    │       │   │    ↓                     │
│   ↓                          │       │   │ retrieval_planner        │
│ extraction                   │       │   │    ↓                     │
│   ↓                          │       │   │ answer_agent             │
│ benchmark                    │       │   │    ↓                     │
│   ↓                          │       │   │ citation_critic          │
│ readiness                    │       │   │    ├─ repair -> answer   │
│   ↓                          │       │   │    └─ accepted -> END    │
│ report                       │       │   ├─ clarification -> END    │
│   ↓                          │       │   └─ analyze_paper -> END    │
│ evidence_critic              │       │                              │
│   ↓                          │       └──────────────────────────────┘
│ report_finalize              │
│   ↓                          │
│ chunk_and_index              │       │   └─ discover -> END         │
│   ├─ next paper -> ingestion │
│   ├─ compare 2+ papers       │
│   └─ END                     │
└──────────────────────────────┘
                       ┌──────────────────────────────┐
                       │       DISCOVERY GRAPH         │
                       │                              │
                       │ research_strategist          │
                       │   ↓                          │
                       │ deterministic searcher       │
                       │   ↓                          │
                       │ selection_advisor            │
                       │   ↓                          │
                       │ END -> session.phase=select  │
                       └──────────────────────────────┘
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
- Artifact storage for PDFs, page images, formulas, and large outputs is not
  implemented yet.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication, rate limiting, and deployment hardening are future work.
