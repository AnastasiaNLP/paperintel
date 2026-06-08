# Distributed Resilience Runbook

PaperIntel coordinates arXiv and Semantic Scholar access through Postgres-backed
provider state when services are created by the application factory.

## Failure Classes

| failure_class | Meaning | Typical worker action |
| --- | --- | --- |
| `rate_limited` | Provider returned 429 or quota pressure. | Retry after normal backoff or `Retry-After`. |
| `provider_unavailable` | Provider 5xx, transport failure, or open shared breaker. | Retry while attempts remain. |
| `provider_timeout` | Provider timed out. | Retry while attempts remain. |
| `dependency_unavailable` | Required infrastructure dependency is unavailable. | Retry while attempts remain. |
| `provider_not_found` | Requested external resource does not exist. | Fail fast. |
| `invalid_input` | Deterministic local validation failure. | Fail fast. |

## Inspect Limiter State

```sql
select provider, operation, next_allowed_at, updated_at
from provider_rate_limits
order by provider, operation;
```

Expected keys:

- `arxiv/api` for arXiv metadata, download, and discovery search.
- `semantic_scholar/api` for Semantic Scholar enrichment.

`next_allowed_at` in the future means callers have reserved future provider
slots. This is normal under concurrent worker load.

## Inspect Breaker State

```sql
select
  provider,
  operation,
  state,
  failure_count,
  failure_threshold,
  open_until,
  last_failure_class,
  updated_at
from provider_circuit_breakers
order by provider, operation;
```

`open` means calls fail fast internally until `open_until`. Public/job output
should show `failure_class=provider_unavailable` with neutral text. `half_open`
means one shared probe is in flight; other processes should still receive open
decisions until that probe records success or failure.

## Operational Response

- For `rate_limited`, reduce concurrency or wait. 429 does not open the circuit
  breaker; it feeds retry and limiter behavior.
- For `provider_unavailable` with an open breaker, wait until `open_until`.
  Repeated manual retries before that point should not help.
- For repeated `provider_timeout`, check network/provider health and worker
  concurrency.
- For `dependency_unavailable`, inspect the named infrastructure dependency
  first, usually Postgres, Qdrant, or blob storage.

## Worker Retries

Worker job error JSON includes:

- `failure_class`
- `retryable`
- `retry_after_seconds` when a retry delay hint is active

`retryable=true` means the current job will actually be retried. Terminal failed
jobs after exhausting attempts persist `retryable=false`.

Example retried provider failure:

```json
{
  "error": "exception",
  "failure_class": "provider_unavailable",
  "retryable": true,
  "retry_after_seconds": 30.0,
  "message": "Provider is temporarily unavailable"
}
```

## Verification Commands

Run unit checks without auto-loaded local plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_provider_policy.py tests/unit/test_provider_rate_limiter.py tests/unit/test_provider_circuit_breaker.py tests/unit/test_health.py tests/unit/test_workflow_worker.py -q
```

Run Postgres-backed limiter, breaker, and worker retry proofs:

```bash
PAPERINTEL_TEST_DATABASE_URL='postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/integration/test_postgres_repositories.py -k 'provider_rate_limit or provider_circuit_breaker or retry_after or retry_respects_schedule or terminal_failure_retryable_false or retry_delay_blocks_other_worker_claims' -q
```

Run the app-factory health proof for the shared provider resilience store:

```bash
PAPERINTEL_TEST_DATABASE_URL='postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/integration/test_paperintel_service_integration.py -k 'app_factory_health_includes_provider_resilience_store' -q
```

Run migration smoke after schema changes:

```bash
PAPERINTEL_TEST_DATABASE_URL='postgresql+psycopg://paperintel:dev_password@localhost:5432/paperintel' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/integration/test_postgres_migration_smoke.py -q
```

## Development Notes

Do not run multiple Alembic-backed pytest processes against the same Postgres
test database at the same time. The fixtures downgrade/upgrade the schema, so
parallel processes can collide during migration setup.

In development, reset provider breaker state by deleting the relevant row:

```sql
delete from provider_circuit_breakers
where provider = 'arxiv' and operation = 'api';
```

Use this only for local/test recovery. Production reset should be accompanied by
checking upstream provider health and current worker concurrency.
