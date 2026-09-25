# Preview Deployment

This checklist is for a temporary hosted preview where a small group can try
PaperIntel.

For a single Hetzner Cloud server, follow the
[Terraform setup and deployment guide](../deploy/terraform/hetzner/README.md).
On a fresh host, `deploy/preview/bootstrap.sh` securely prompts for provider
keys, creates the private `.env`, builds the stack, and waits for HTTPS health.

## Required Services

- Web process: `uvicorn api.rest.main:app --host 0.0.0.0 --port $PORT`
- Worker process: `python -m workers`
- Cleanup process: `python -m workers.blob_cleanup_worker --interval-seconds 3600`
- Postgres database
- Qdrant vector database
- S3-compatible object storage such as MinIO, AWS S3, or Cloudflare R2

The repository Compose profile runs the API behind Caddy with automatic HTTPS,
the workflow worker, hourly blob cleanup worker, migrations, Postgres, Qdrant,
and MinIO together. Point the preview hostname to the server and allow inbound
TCP ports 80 and 443 (plus UDP 443 for HTTP/3) before starting it. Container
JSON logs are rotated at 10 MB with five files retained per service.

```bash
ANTHROPIC_API_KEY=... \
OPENAI_API_KEY=... \
PAPERINTEL_API_AUTH_TOKEN="$(openssl rand -hex 32)" \
PAPERINTEL_CORS_ALLOW_ORIGINS=https://preview.example.com \
PREVIEW_DOMAIN=preview.example.com \
docker compose --profile app up -d --build
```

The UI is served at `https://preview.example.com/`. The API listens only on
localhost port 8000; Caddy is the only public application entry point. Data
services also bind to localhost ports for inspection. For a cloud host, use
managed Postgres, Qdrant, and object storage where available, and deploy API
and worker as separate processes from the same image.

For a host that manages processes separately, apply migrations before starting
the web and worker processes:

```bash
python -m alembic upgrade head
```

## Required Environment

Set these on the host:

```text
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
POSTGRES_URL=...
QDRANT_URL=...
QDRANT_COLLECTION=paper_chunks
BLOB_STORAGE_ENABLED=true
BLOB_S3_ENDPOINT_URL=...
BLOB_S3_REGION=...
BLOB_S3_BUCKET=...
BLOB_S3_ACCESS_KEY_ID=...
BLOB_S3_SECRET_ACCESS_KEY=...
PAPERINTEL_API_AUTH_TOKEN=<long random preview token>
PAPERINTEL_CORS_ALLOW_ORIGINS=https://your-preview-ui.example
LANGCHAIN_TRACING_V2=false
```

`LANGCHAIN_API_KEY` can be set to `disabled` when tracing is off. Configure
`POSTGRES_URL` in both the API and worker environments; Alembic reads this same
value when applying migrations.

Use `PAPERINTEL_API_AUTH_TOKEN` for temporary public previews. When it is set,
API endpoints require:

```text
Authorization: Bearer <token>
```

`/health`, `/docs`, `/redoc`, and `/openapi.json` stay open so the preview can
be inspected and monitored.

## Smoke Check

After deployment, run the authenticated smoke check from a machine that can
reach the public preview. It creates a session, runs paid discovery and one
paper analysis, and persists the resulting workspace; use it once per clean
deployment, not as a frequent health probe:

```bash
PAPERINTEL_BASE_URL=https://your-host.example \
PAPERINTEL_API_AUTH_TOKEN=... \
.venv/bin/python deploy/preview/smoke_test.py
```

The script verifies service health, discovery candidates, selection, async job
execution, persisted report, and a grounded QA answer. It can take up to 20
minutes by default; set `PAPERINTEL_SMOKE_TIMEOUT_SECONDS` to change the limit.
For a cheap non-mutating availability probe, use `curl
https://your-host.example/health` instead.

The response should be HTTP 200 with these checks healthy or configured:

- `postgres=ok`
- `provider_resilience_store=ok`
- `qdrant=ok`
- `llm_provider=configured`
- `openai_embeddings=configured`
- `blob_store=ok`

Create a session with the preview token:

```bash
curl -X POST https://your-host.example/sessions \
  -H "content-type: application/json" \
  -H "authorization: Bearer <token>" \
  -d '{"persona":"engineer"}'
```

For public testers, prefer async analysis endpoints plus the worker process so
long paper analysis does not tie up HTTP requests.

## Rollback and Data

Keep the previously deployed Git revision available. To stop the preview while
preserving its data, run `docker compose --profile app down` from the project
directory; never add `-v` unless deleting all preview data is intentional. To
roll back application code, check out the previous revision and rebuild with
`docker compose --profile app up -d --build`. Database migrations are forward
only for this preview: do not downgrade after writes; restore a database backup
before attempting a schema rollback. The Hetzner Terraform starter enables
daily whole-server backups (seven restore points); an application-level dump
and a restore drill are still required before treating tester data as durable.
