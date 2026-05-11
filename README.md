# PaperIntel

PaperIntel is a research intelligence system for AI/ML papers. It is designed to
help engineers, researchers, and technical leads analyze known papers today and
grow into persistent discovery, comparison, and conversational QA workflows.

The current implementation has two strong foundations:

- a known-paper analysis pipeline for arXiv URLs, PDFs, and URL batches
- a production data foundation with sessions, turns, structured errors,
  AgentRun tracing, runtime policies, Alembic migrations, and Postgres-backed
  repositories

Discovery, retrieval, full conversational QA, jobs, outbox, cache, and transport
APIs are planned but not implemented yet.

---

## Recent Milestones

- **Production Data Foundation (closed):** ChatHandler with
  in-memory and Postgres backends, Alembic migrations, repository pattern
  with Pydantic ↔ ORM mappers, dependency injection for storage.
- **Production Agent Contract (closed):** AgentRun lifecycle in
  graph state, AgentRuntimePolicy schema with selective enforcement,
  persistence seam (Noop / InMemory / Postgres), applied to `report` and
  `evidence_critic` nodes. Checkpoint serialization verified.
- **Earlier — Repository hygiene:** structured errors, runnable baseline,
  test layout (offline vs live), LLM provider abstraction.

---

## Current Status

Implemented analysis pipeline:

- arXiv URL ingestion
- PDF parsing with PyMuPDF
- arXiv metadata lookup
- Semantic Scholar enrichment
- method extraction
- benchmark extraction
- production-readiness assessment
- engineer report generation
- Evidence Critic review after report generation
- report finalization into `PaperSlot`
- multi-paper comparison
- LangGraph orchestration
- batch processing for multiple paper URLs
- LangGraph checkpointing with both in-memory (MemorySaver, default) and
  Postgres (PostgresSaver, optional) backends; graph state including
  AgentRun and policy snapshots is serialization-safe (verified through
  checkpoint round-trip tests)

Implemented production data foundation:

- explicit `ChatHandler.create_session(...)`
- explicit `ChatHandler.handle_message(session_id, message)`
- `Session`, `Turn`, `HandlerResult`, and `GraphInvocationResult` models
- `Session.persona` field (engineer / researcher / techlead) stored but
  not yet consumed by the analysis flow (persona-aware behavior planned
  for agentic layer rollout)
- `SessionStore` protocol
- `InMemorySessionStore`
- `PostgresSessionStore`
- `PostgresAgentRunPersistence`
- `PostgresStructuredErrorRepository`
- `storage/mappers.py` with bidirectional Pydantic ↔ ORM mappers
- `clear_foundation_tables` utility for test cleanup
- SQLAlchemy 2.0 ORM models
- Alembic initial schema migration
- tables: `sessions`, `turns`, `agent_runs`, `structured_errors`
- `api/app_factory.create_chat_handler(...)` for application bootstrap
- manual and automated Postgres smoke tests

Implemented controlled agent contract:

- `AgentRun` lifecycle in graph state
- `AgentRuntimePolicy` defaults and per-call overrides
- `report_agent` records success, repair, and error runs
- `evidence_critic_agent` records pass-through, downgrade, and skipped runs
- inter-agent traceability through `input_refs`
- policy snapshots in `AgentRun.details`
- report policy warning when `llm_call_count` exceeds `max_tool_calls`
- checkpoint serialization coverage for real `AgentRun` objects

**Note on scope:** the production agent contract is currently implemented for
two nodes: `report` (wrapped workflow processor with real `llm_call_count`
enforcement) and `evidence_critic` (first production-shaped critic, MVP
deterministic mode). Other LLM-assisted nodes — extraction, benchmark,
readiness, comparator — remain analysis pipeline processors. Wrapping them in
the full AgentRun / AgentRuntimePolicy contract is planned for the agentic
layer rollout.

Planned layers:

- FastAPI, Gradio, and MCP transport layer
- real `ConversationGraph`
- intent routing and reference resolution
- retrieval with Qdrant
- artifact storage for PDFs, raw text, page images, formulas, and agent outputs
- paper cache with versioning
- pg_boss jobs
- outbox events
- session budgets
- discovery agents (Research Strategist, Searcher, Selection Advisor)
- QA agents (Intent Router, Evidence Retrieval Planner, Answer, Citation Critic)
- comparison analyst and synthesis agents
- DeepEval, LangSmith, Prometheus, and Grafana observability

---

## Implemented Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                      API / HANDLER LAYER                         │
│                                                                  │
│  ChatHandler (constructed via app_factory.create_chat_handler)   │
│  - create_session                                                │
│  - handle_message                                                │
│  - writes user turn before graph call                            │
│  - writes assistant turn after graph call                        │
│  - graph failure -> StructuredError                              │
│  - passes session_id + AgentRunPersistence via RunnableConfig    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CURRENT GRAPH PIPELINE                        │
│                                                                  │
│  supervisor                                                      │
│      ↓                                                           │
│  ingestion -> extraction -> benchmark -> readiness               │
│      ↓                                                           │
│  report -> evidence_critic -> report_finalize                    │
│      ↓                                                           │
│      ├─ next paper in batch -> ingestion                         │
│      └─ 2+ completed papers -> comparator -> END                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SERVICE / STORAGE LAYER                      │
│                                                                  │
│  SessionStore                                                    │
│    - InMemorySessionStore                                        │
│    - PostgresSessionStore                                        │
│                                                                  │
│  AgentRunPersistence                                             │
│    - NoopAgentRunPersistence                                     │
│    - InMemoryAgentRunPersistence                                 │
│    - PostgresAgentRunPersistence                                 │
│                                                                  │
│  PostgresStructuredErrorRepository                               │
│  SQLAlchemy mappers (Pydantic ↔ ORM)                             │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                            POSTGRES                              │
│                                                                  │
│  Alembic-managed schema                                          │
│  - sessions                                                      │
│  - turns                                                         │
│  - agent_runs                                                    │
│  - structured_errors                                             │
└──────────────────────────────────────────────────────────────────┘
```

This is not yet the final discovery/QA system. It is the stateful production
foundation that the future conversation and retrieval layers will use.

---

## Target Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                         TRANSPORT LAYER                            │
│                                                                    │
│     FastAPI                    Gradio                   MCP Server  │
│       │                         │                          │        │
│       └─────────────────────────┴──────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│                       API / Chat Handler                           │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                  JOB / ORCHESTRATION LAYER                         │
│                                                                    │
│   pg_boss job queue + Outbox Pattern                               │
│                                                                    │
│   ┌────────────────────────────┐    ┌────────────────────────────┐ │
│   │     ConversationGraph       │    │   ResearchWorkflowGraph    │ │
│   │                            │    │                            │ │
│   │ load_session               │    │ start / resume             │ │
│   │ intent_router (agent)      │───►│ discovery_team             │ │
│   │ qa_team                    │    │ selection_gate             │ │
│   └──────────────┬─────────────┘    │ analysis_batch + critic    │ │
│                  │                  │ comparison_team            │ │
│                  │                  └──────────────┬─────────────┘ │
└──────────────────┼──────────────────────────────────┼──────────────┘
                   │                                  │
                   ▼                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    AGENTIC LAYER (controlled)                       │
│                                                                    │
│ Analysis Review:        Discovery Team:        Comparison:          │
│ - Evidence Critic       - Research Strategist  - Comparison Analyst │
│                         - Searcher             - Synthesis Agent    │
│                         - Selection Advisor                         │
│                                                                    │
│ QA Team:                                                            │
│ - Intent Router with reference resolution                           │
│ - Evidence Retrieval Planner                                        │
│ - Answer Agent                                                      │
│ - Citation Critic                                                   │
│                                                                    │
│ Auxiliary:                                                          │
│ - Selection Negotiator                                              │
└──────────────────┬──────────────────────────────────┬──────────────┘
                   │                                  │
                   ▼                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                           SERVICE LAYER                             │
│                                                                    │
│   SessionRepository                                                 │
│   ArtifactRepository                                                │
│   PaperCacheRepository                                              │
│   AgentRunRepository                                                │
│   RetrievalLayer                                                    │
│   SelectionParser                                                   │
│   StructuredErrorService                                            │
│   SessionBudgetService                                              │
│   Metrics / Tracing hooks                                           │
└──────────────────┬──────────────────────────────────┬──────────────┘
                   │                                  │
                   ▼                                  ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│       SESSION STORAGE         │       │       ARTIFACT STORAGE        │
│       PostgreSQL              │       │       S3 / MinIO              │
│                              │       │                              │
│ sessions                     │       │ PDFs                         │
│ turns                        │       │ raw text                     │
│ jobs                         │       │ page images                  │
│ outbox_events                │       │ table images                 │
│ candidates                   │       │ formula/page renders         │
│ workspaces                   │       │ extracted large blobs        │
│ chunks metadata              │       │ agent_outputs (large)        │
│ artifacts metadata           │       │                              │
│ comparisons                  │       │                              │
│ paper_cache metadata         │       │                              │
│ agent_runs                   │       │                              │
│ critic_reviews               │       │                              │
│ structured_errors            │       │                              │
│ session_budgets              │       │                              │
└──────────────────────────────┘       └──────────────────────────────┘

┌──────────────────────────────┐
│            QDRANT             │
│        Vector Storage         │
│                              │
│ paper chunks                 │
│ equation contexts            │
│ table contexts               │
│ session-scoped filters       │
└──────────────────────────────┘
```

The target architecture is intentionally broader than the current code. The
current implemented subset is the known-paper analysis pipeline plus the durable
session and agent-run foundation.

---

## Implemented Processing Pipeline

The implemented graph path is the known-paper analysis flow inside the future
`ResearchWorkflowGraph`.

```text
Known arXiv URL / PDF / batch URLs
          │
          ▼
┌──────────────────────┐
│ Supervisor / Router   │
│ validates stage       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Ingestion Agent       │
│ arXiv, S2, PDF parse, │
│ abstract fallback     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Extraction Agent      │
│ method, novelty,      │
│ components, limits    │
└──────────┬───────────┘
           │
           ├── low confidence ──► Human Review Gate
           │                          │
           │                          ▼
           └────────────────────► Benchmark Agent
                                  │
                                  ▼
┌──────────────────────┐
│ Benchmark Agent       │
│ tables, metrics,      │
│ fallback text context │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Readiness Agent       │
│ GitHub, HF resources, │
│ dependencies, maturity│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Report Agent          │
│ EngineerReport,       │
│ markdown, AgentRun    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Evidence Critic       │
│ pass-through,         │
│ downgrade, skipped,   │
│ AgentRun input_refs   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Report Finalize       │
│ stores PaperSlot      │
│ and resets scratch    │
└──────────┬───────────┘
           │
           ├── next paper in batch ──► Ingestion Agent
           │
           └── 2+ papers complete ──► Comparator Agent ──► END
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Comparator Agent      │
                           │ benchmark matrix,     │
                           │ trade-offs, winner    │
                           └──────────────────────┘
```

The orchestration is deterministic. The analysis steps are LLM-assisted, and the
production-shaped agent contract is currently implemented for `report` and
`evidence_critic`.

---

## Project Structure

```text
agents/
  Workflow processors:
    ingestion, extraction, benchmark, readiness, report, comparator
  Production-shaped agents:
    evidence_critic
  Graph gates and finalizers:
    human_review, report_finalize, paper_failure_finalize, supervisor
  Support modules:
    agent_run_recorder, error_utils, llm_provider

api/             ChatHandler, app_factory, SessionStore protocol, InMemorySessionStore
storage/         SQLAlchemy ORM models, mappers, Postgres repositories
alembic/         Database migrations
docs/            Engineering documentation
tools/           External API clients (arXiv, S2, GitHub, HF) and PDF parser
models/          Pydantic schemas: state, session, agent_runs, policies, errors
config/          Settings and LLM prompt files
tests/           unit/ + integration/ (offline + db-marked) + live/
graph.py         Main LangGraph assembly with serializer allowlist
```

**Note on naming:** the `agents/` directory contains both workflow processors
and production-shaped agents. By the production rule — an agent makes real
decisions that pipeline cannot safely hardcode — not every file in `agents/`
is an agent. Renaming is intentionally deferred: it would require import churn
across the codebase and is low priority compared to feature work.

---

## Documentation

- [docs/AGENT_CONTRACT.md](docs/AGENT_CONTRACT.md) — how to wrap a node
  in the AgentRun lifecycle and AgentRuntimePolicy contract

---

## Requirements

- Python 3.11+
- API keys for the selected LLM provider (Anthropic or OpenAI) — required
  for LLM-assisted nodes
- (Optional) PostgreSQL 16 — for durable session storage and Postgres-
  backed graph checkpointing. Not required for offline tests or default
  in-memory usage.

---

## Environment

Create `.env` from `.env.example`.

Common variables:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
LANGCHAIN_API_KEY
LANGCHAIN_TRACING_V2
LANGCHAIN_PROJECT
GITHUB_TOKEN
LLM_PROVIDER
OPENAI_MODEL
DATABASE_URL                       # production Postgres connection (optional)
PAPERINTEL_TEST_DATABASE_URL       # test Postgres for db-marked tests (optional)
```

Use `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai`.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using project metadata directly:

```bash
pip install -e .
```

---

## Database

Start local Postgres:

```bash
docker compose up -d postgres
```

Run migrations:

```bash
.venv/bin/python -m alembic upgrade head
```

Manual inspection:

```bash
docker compose exec postgres psql -U paperintel -d paperintel
```

Useful SQL:

```sql
select id, persona, phase, original_query from sessions;
select role, content, intent, referenced_paper_ids, artifact_refs from turns order by created_at;
select id, agent_name, status, output_ref, details_json from agent_runs;
select id, code, message, severity from structured_errors;
```

---

## Tests

The test suite has three categories:

- **Offline (default):** unit + integration tests, no network and no
  database required.
- **DB-marked:** Postgres migration, repository, and handler tests. Opt-in
  via `PAPERINTEL_TEST_DATABASE_URL`.
- **Live:** tests requiring real LLM credentials and external APIs.
  Opt-in via `-m live`.

Offline tests must not require external API calls or a database URL.

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Run unit tests:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/unit
```

Run integration tests without enabling DB smoke tests:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/integration
```

Enable Postgres-backed tests explicitly:

```bash
export PAPERINTEL_TEST_DATABASE_URL="postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel"

LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/integration/test_postgres_migration_smoke.py
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/integration/test_postgres_repositories.py
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/integration/test_postgres_chat_handler.py
```

Live tests require network access and real credentials:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/live -m live
```