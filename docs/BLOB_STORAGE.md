# Blob Storage

PaperIntel stores uploaded and locally analyzed PDFs in S3-compatible object
storage. The default local backend is MinIO from `docker-compose.yml`.

## Local Setup

Start MinIO with the other local dependencies:

```bash
docker compose up -d postgres qdrant minio
```

The MinIO API is available at `http://localhost:9000`. The development console
is available at `http://localhost:9001`.

Configure the backend through `.env`:

```text
BLOB_STORAGE_ENABLED=true
BLOB_S3_ENDPOINT_URL=http://localhost:9000
BLOB_S3_REGION=us-east-1
BLOB_S3_BUCKET=paperintel
BLOB_S3_ACCESS_KEY_ID=paperintel
BLOB_S3_SECRET_ACCESS_KEY=paperintel_dev_password
```

When blob storage is enabled, application startup verifies that the bucket
exists and creates it if necessary. `/health` reports `blob_store=ok` only when
the configured backend is ready.

## Storage Model

PDFs use content-addressed object keys:

```text
papers/sha256/<first-two-hex-characters>/<full-sha256>.pdf
```

Uploading the same PDF more than once reuses the existing object. Postgres
stores one `blob_artifacts` registry row for the physical object and separate
`blob_references` rows for sessions and paper workspaces that use it.

REST multipart uploads still use a short-lived local temp file while the
request is processed. MCP local-path analysis reads the trusted server-local
file. In both cases, the PDF content is persisted to blob storage before the
analysis pipeline runs against a verified temporary materialization.

## Retention

PDF blobs use durable retention by default. The cleanup engine expires stale
client-upload staging records and removes unreferenced TTL blob artifacts. It
does not delete durable PDF blobs that still have active session, workspace, or
workflow-job references.

Do not delete S3 or MinIO objects without reconciling Postgres metadata.
Deleting an object directly can leave an orphan `blob_artifacts` row.

Cleanup is bounded and idempotent. `BlobCleanupService.run_once(dry_run=True)`
lists candidates without deleting objects or changing Postgres state. A normal
run deletes the physical object first and only then marks the Postgres upload or
blob artifact cleaned up. If object deletion fails, the registry row remains
active for a later retry.

## Current Limits

- PDF uploads are loaded into memory before upload. The REST API limits uploads
  to 50 MB.
- Streaming uploads are not implemented.
- Async PDF workflow jobs use durable blob references. Canceling a queued job
  does not delete the underlying PDF blob.
- Cleanup is available as a service API and manual worker command, but there is
  no scheduler or REST endpoint yet.
