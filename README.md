# PaperIntel

PaperIntel is a research intelligence system for AI/ML papers. It analyzes
arXiv papers, indexes them for retrieval, and answers questions with citations
from the paper.

It is built for engineers, researchers, and technical leads who want to move
from "I have a paper URL" to "I understand the method, evidence, limitations,
and implementation implications" without losing grounding in the source text.

## What It Does

- Analyzes arXiv papers and PDFs.
- Extracts method, benchmarks, implementation readiness, and engineering notes.
- Chunks and indexes analyzed papers into Postgres + Qdrant.
- Answers questions about analyzed papers with citations.
- Discovers recent papers for a topic, ranks candidates, and lets the user
  select papers by display number.
- Analyzes selected discovery candidates, indexes them, and makes them
  available for citation-backed QA.
- Produces a batch comparison report when multiple selected papers are analyzed
  together.
- Persists analysis workspaces and batch comparison artifacts so they can be
  reloaded without re-running analysis.
- Synthesizes active papers on demand through retrieval-backed QA with
  citations.
- Uses an adversarial Citation Critic with bounded repair to reduce unsupported
  confident claims.
- Supports persona-aware answers: `engineer`, `researcher`, and `techlead`.
- Exposes both a REST API and an MCP server.

## Quick Start

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for the full setup.

Short version:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add ANTHROPIC_API_KEY and OPENAI_API_KEY to .env

docker compose up -d postgres qdrant
.venv/bin/python -m alembic upgrade head

.venv/bin/python -m dotenv run -- \
  .venv/bin/uvicorn api.rest.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## REST Example

```bash
SESSION_ID=$(
  curl -s -X POST http://127.0.0.1:8000/sessions \
    -H 'content-type: application/json' \
    -d '{"persona":"engineer"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])"
)

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/analyze" \
  -H 'content-type: application/json' \
  -d '{"paper_url":"https://arxiv.org/abs/1706.03762"}'

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"What is the main contribution of this paper?"}'

curl -s "http://127.0.0.1:8000/sessions/$SESSION_ID/workspaces"
```

Discovery workflow:

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/discover" \
  -H 'content-type: application/json' \
  -d '{"topic":"Find recent papers about retrieval augmented generation"}'

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/select" \
  -H 'content-type: application/json' \
  -d '{"selection":"use 1 and 3"}'

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/analyze-selected"

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/ask" \
  -H 'content-type: application/json' \
  -d '{"question":"What is the main contribution of the selected paper?"}'

curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/synthesize" \
  -H 'content-type: application/json' \
  -d '{"prompt":"Compare the selected papers for implementation trade-offs."}'

curl -s "http://127.0.0.1:8000/sessions/$SESSION_ID/comparison"
```

For a runnable script, see [examples/rest_smoke.py](examples/rest_smoke.py).

## MCP

PaperIntel includes a local MCP server for Claude Desktop and other MCP
clients:

```bash
.venv/bin/python -m mcp_server.server
```

See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for Claude Desktop configuration and
example prompts.

## Architecture

The current system has four main layers:

```text
REST / MCP
    ↓
PaperIntelService
    ↓
ChatHandler
    ├─ analysis graph: ingest -> extract -> report -> critic -> chunk/index
    ├─ conversation graph: route -> retrieve -> answer -> citation critic
    └─ discovery graph: plan -> arXiv search -> rank -> selection advice
```

Full architecture details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md) — local setup and first paper.
- [docs/API.md](docs/API.md) — REST API usage patterns.
- [docs/MCP_SETUP.md](docs/MCP_SETUP.md) — MCP server setup.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common issues.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — implemented architecture.
- [docs/AGENT_CONTRACT.md](docs/AGENT_CONTRACT.md) — AgentRun and policy contract.
- [docs/CHUNKING_STRATEGY.md](docs/CHUNKING_STRATEGY.md) — retrieval chunking decisions.

## Requirements

- Python 3.11+
- Anthropic API key for default LLM reasoning
- OpenAI API key for embeddings
- Docker for local Postgres and Qdrant

## Tests

Default non-live suite:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -m 'not live'
```

Current non-live coverage:

- 542 passing unit and integration tests in the default non-live/non-DB profile
- 18 DB-marked tests skipped unless `PAPERINTEL_TEST_DATABASE_URL` is set
- live QA and discovery tests requiring real LLM credentials and local services

Live QA smoke:

```bash
docker compose up -d postgres qdrant
.venv/bin/python -m dotenv run -- env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/pytest -s tests/live/test_qa_conversation_live.py
```

The live QA test is expected to take roughly 90 seconds on a local Docker
Postgres/Qdrant stack.

Live discovery-to-QA smoke:

```bash
docker compose up -d postgres qdrant
.venv/bin/python -m dotenv run -- env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/pytest -s tests/live/test_discovery_to_qa_live.py
```

Discovery live tests depend on the public arXiv API. They skip instead of
failing when arXiv rate-limits all search queries.

Recent live discovery-to-QA verification completed successfully in about 92
seconds: discovery returned 10 candidates, one selected paper was analyzed and
indexed, QA returned a cited answer with 3 citations, and all recorded agent
runs completed without failures.

## Current Limitations

- REST and MCP analysis/discovery calls are synchronous.
- Discovery currently searches arXiv only.
- Discovery plus comparison/synthesis is implemented as an MVP: discovery,
  shortlist selection, selected-paper analysis, batch comparison artifacts, and
  retrieval-backed synthesis are working.
- Dedicated `comparison_analyst` and `synthesis_agent` components are deferred
  until the current artifact layer is used as their input contract. The durable
  workspaces now exist; agent design, prompt contracts, and evaluation are a
  separate follow-up.
- `agents/comparator.py` is a known transitional component: it remains the
  batch analysis comparator for multi-paper analysis and is expected to migrate
  into a future `comparison_analyst` path built on persisted artifacts.
- Artifact persistence is intentionally narrow: Postgres stores finalized
  reports, method extraction, benchmarks, readiness results, and comparison
  reports. S3/object storage, paper cache versioning, outbox/job processing,
  and PDF asset storage are separate later work.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication, rate limiting, and deployment hardening are future work.

## License

MIT. See [LICENSE](LICENSE).
