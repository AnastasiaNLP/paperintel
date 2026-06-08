# PaperIntel

PaperIntel is a research intelligence system for AI/ML papers. It analyzes
arXiv papers, indexes them for retrieval, and answers questions with citations
from the paper.

It is built for engineers, researchers, and technical leads who want to move
from "I have a paper URL" to "I understand the method, evidence, limitations,
and implementation implications" without losing grounding in the source text.

## What It Does

- Analyzes arXiv papers, uploaded PDFs through REST, and trusted local PDFs through MCP.
- Extracts method, benchmarks, implementation readiness, and engineering notes.
- Chunks and indexes analyzed papers into Postgres + Qdrant.
- Answers questions about analyzed papers with citations.
- Discovers recent papers for a topic, ranks candidates, and lets the user
  select papers by display number.
- Analyzes selected discovery candidates, indexes them, and makes them
  available for citation-backed QA.
- Produces a batch comparison report when multiple selected papers are analyzed
  together.
- Creates request-driven comparison artifacts with `comparison_analyst`.
- Synthesizes active papers with a dedicated `synthesis_agent` over durable
  workspaces, optionally using the latest comparison as context.
- Persists analysis workspaces and comparison artifacts so they can be reloaded
  without re-running analysis.
- Includes deterministic artifact checks, a 30-paper golden dataset, CA
  structural checks, and non-gating G-Eval rubric gauges.
- Uses an adversarial Citation Critic with bounded repair to reduce unsupported
  confident claims.
- Supports persona-aware answers: `engineer`, `researcher`, and `techlead`.
- Exposes both a REST API and an MCP server.
- Supports Postgres-backed async analysis jobs with a separate worker process.
- Caches arXiv metadata and degrades gracefully when paper metadata enrichment fails.
- Persists analyzed PDFs in S3-compatible object storage with content-hash deduplication.

## Quick Start

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for the full setup.

Short version:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add ANTHROPIC_API_KEY and OPENAI_API_KEY to .env

docker compose up -d postgres qdrant minio
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

Local PDF upload through REST:

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/analyze-pdf" \
  -F "file=@/absolute/path/to/paper.pdf;type=application/pdf" \
  -F "paper_id=local-paper-1" \
  -F "skip_arxiv_metadata_fetch=true"
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

## Evaluation

PaperIntel includes an evaluation MVP for persisted artifacts. The deterministic
runner checks benchmark rows, readiness fields, and keyword coverage over
exported `PaperWorkspace` JSONL files. A separate judge runner can score
engineer-report rubrics manually through the configured LLM provider.

The repository includes both a fast 5-paper seed and a schema-clean 30-paper
golden dataset published on Hugging Face:
[AIAnastasia/arxiv-papers](https://huggingface.co/datasets/AIAnastasia/arxiv-papers).

See [evaluation/README.md](evaluation/README.md) for commands, scope, and known
limitations. Deterministic checks are suitable for CI-style gating; LLM-judge
scores are non-deterministic and treated as manual/scheduled quality signals.

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
- [docs/ASYNC_JOBS.md](docs/ASYNC_JOBS.md) — workflow job queue and worker operations.
- [docs/RESILIENCE.md](docs/RESILIENCE.md) — arXiv/Semantic Scholar cache,
  limits, breakers, and fallback behavior.
- [docs/PAPER_CACHE.md](docs/PAPER_CACHE.md) — cross-session paper analysis
  reuse contract, public metadata, limitations, and verification checklist.
- [docs/BLOB_STORAGE.md](docs/BLOB_STORAGE.md) — MinIO/S3 PDF storage, deduplication,
  retention, and current limits.
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

- unit and integration tests cover service, graph, REST, MCP, storage, and evaluation paths
- DB-marked tests require `PAPERINTEL_TEST_DATABASE_URL`
- live QA, discovery, blob-storage, and job tests require real provider credentials and local services

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

- REST and MCP still expose synchronous analysis/discovery calls. Async URL
  analysis is available through workflow jobs and a separate worker process.
  Local PDF analysis is available through synchronous calls and async PDF jobs.
- Discovery currently searches arXiv only.
- Discovery plus comparison/synthesis is implemented: discovery, shortlist
  selection, selected-paper analysis, batch comparison artifacts, request-driven
  comparison, and dedicated synthesis are working.
- `comparison_analyst` and `synthesis_agent` are production-shaped agents over
  durable paper workspaces.
- `agents/comparator.py` remains as the legacy batch-analysis comparator and
  writes comparison artifacts with `producer="batch_comparator"`;
  request-driven comparisons use `producer="comparison_analyst"`.
- Postgres stores finalized reports, method extraction, benchmarks, readiness
  results, comparison reports, workflow jobs, and blob metadata. S3-compatible
  storage persists PDFs with content-hash deduplication. Completed paper
  analysis can be reused across sessions by cloning ready workspaces and
  retrieval chunks. A separate `paper_cache` table, advanced job
  scheduling/retries, and scheduled cleanup jobs remain later work.
- Critic conflict resolution is deferred until structured claim provenance is
  added.
- Authentication and deployment hardening are future work. arXiv and Semantic
  Scholar use Postgres-backed provider rate limiting and circuit-breaker state
  when services are created through the application factory.

## License

MIT. See [LICENSE](LICENSE).
