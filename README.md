# PaperIntel

PaperIntel is a research intelligence system for AI/ML papers. It is designed to help engineers and technical leads discover, analyze, compare, and discuss research papers in a persistent research session.

The current codebase already contains the core paper-analysis pipeline for known arXiv URLs and PDFs. The target architecture extends that core into a production workflow with discovery, conversational QA, persistent sessions, retrieval, controlled multi-agent review, cache, jobs, and observability.

---

## Current Status

Implemented core pipeline:

- arXiv URL ingestion
- PDF parsing with PyMuPDF
- arXiv metadata lookup
- Semantic Scholar enrichment
- method extraction
- benchmark extraction
- production-readiness assessment
- engineer report generation
- multi-paper comparison
- LangGraph orchestration
- PostgreSQL checkpointing fallback
- batch processing for multiple paper URLs

Planned production layers:

- FastAPI, Gradio, and MCP transport layer
- persistent session storage
- artifact storage for PDFs, raw text, page images, formulas, and agent outputs
- Qdrant-backed retrieval
- pg_boss jobs
- outbox events
- structured errors
- session budgets
- controlled multi-agent layer with critics and bounded loops
- DeepEval, LangSmith, Prometheus, and Grafana observability

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
│ - Intent Router (with reference resolution)                         │
│ - Evidence Retrieval Planner                                        │
│ - Answer Agent                                                      │
│ - Citation Critic                                                   │
│                                                                    │
│ Auxiliary:                                                          │
│ - Selection Negotiator                                              │
│                                                                    │
│ All agents have: MAX_ITERATIONS, fallback, AgentRun persistence.    │
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

Cross-cutting:
┌──────────────────────────────┐       ┌──────────────────────────────┐
│            EVAL               │       │        OBSERVABILITY          │
│ DeepEval golden datasets      │       │ LangSmith · Prometheus        │
│ pytest eval suite             │       │ per-agent metrics · Grafana   │
│ critic eval                   │       │                              │
└──────────────────────────────┘       └──────────────────────────────┘
```

---

## Implemented Processing Pipeline

The implemented part of the system is the known-paper analysis path inside the future `ResearchWorkflowGraph`.

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
│ - arXiv ID extraction │
│ - arXiv metadata      │
│ - Semantic Scholar    │
│ - PDF download/parse  │
│ - abstract fallback   │
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
│ engineer report +     │
│ markdown rendering    │
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
           └── 2+ papers complete ──► Comparator Agent
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Comparator Agent      │
                           │ benchmark matrix,     │
                           │ trade-offs, winner    │
                           └──────────────────────┘
```

This pipeline is deterministic in orchestration and LLM-assisted in the analysis steps. It is the foundation for the production `analysis_batch` path.

---

## Project Structure

```text
agents/          LangGraph nodes and LLM workflow components
tools/           External API clients and PDF utilities
models/          Pydantic schemas and graph state
config/          Settings and prompts
tests/           Unit, integration, and live tests
graph.py         Main LangGraph assembly
```

---

## Requirements

- Python 3.11+
- PostgreSQL for checkpointing
- API keys for selected LLM provider and live integrations

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

## Tests

Offline tests should not require external API calls.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit tests/integration
```

Run only unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit
```

Run only integration tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/integration
```

Live tests require network access and real credentials. They should be marked explicitly:

```python
@pytest.mark.live
```

Run live tests explicitly:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/live -m live
```

