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
    └─ conversation graph: route -> retrieve -> answer -> citation critic
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

- 363 passing unit and integration tests
- 12 DB-marked tests skipped unless `PAPERINTEL_TEST_DATABASE_URL` is set
- 1 live QA conversation test requiring real LLM credentials, Postgres, and
  Qdrant

Live QA smoke:

```bash
docker compose up -d postgres qdrant
.venv/bin/python -m dotenv run -- env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/pytest -s tests/live/test_qa_conversation_live.py
```

The live QA test is expected to take roughly 90 seconds on a local Docker
Postgres/Qdrant stack.

## Current Limitations

- Paper discovery agents are not implemented yet.
- REST and MCP analysis calls are synchronous.
- Artifact storage for PDFs, page images, formulas, and large agent outputs is
  not implemented yet.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication, rate limiting, and deployment hardening are future work.

## License

MIT. See [LICENSE](LICENSE).
