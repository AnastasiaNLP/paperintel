# Quickstart

This guide starts PaperIntel locally with Postgres, Qdrant, the REST API, and a
first paper analysis.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

OpenAI is used for embeddings. Anthropic is the default LLM provider for agent
reasoning.

## 3. Start Local Services

```bash
docker compose up -d postgres qdrant
```

Run migrations:

```bash
.venv/bin/python -m alembic upgrade head
```

`alembic.ini` currently uses the local docker-compose Postgres URL. If you
change Postgres connection settings in `.env`, update `alembic.ini` as well
before running migrations.

## 4. Start the REST API

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/uvicorn api.rest.main:app --host 127.0.0.1 --port 8000
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

## 5. Analyze a Paper

In a second terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"persona":"engineer"}'
```

Copy the returned `id`, then analyze a paper:

```bash
curl -s -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/analyze \
  -H 'content-type: application/json' \
  -d '{"paper_url":"https://arxiv.org/abs/1706.03762"}'
```

Analysis is synchronous and can take about a minute for a typical arXiv paper.

Ask a question:

```bash
curl -s -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/ask \
  -H 'content-type: application/json' \
  -d '{"question":"What is the main contribution of this paper?"}'
```

## 6. Optional: MCP

To use PaperIntel through an MCP client such as Claude Desktop, see
[MCP_SETUP.md](MCP_SETUP.md).

## 7. Tests

Run the default non-live suite:

```bash
LANGCHAIN_TRACING_V2=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -m 'not live'
```

Live tests require external services and credentials. See the testing section in
the main README for details.
