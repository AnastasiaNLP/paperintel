# External Dependency Resilience

PaperIntel depends on public academic services and LLM/vector providers. This
document describes the current protections around arXiv and Semantic Scholar,
which are the two external paper-metadata dependencies most likely to rate-limit
or temporarily fail.

## arXiv Metadata

PaperIntel stores successful arXiv metadata lookups in the global Postgres table
`arxiv_metadata_cache`. The cache is keyed only by `arxiv_id` and is not tied to
a session. This makes metadata reusable across users and sessions.

Cache policy:

- A successful cache hit returns cached metadata and does not call arXiv.
- There is no TTL refresh in the current implementation. arXiv metadata is
  treated as immutable enough for this use case.
- Future `force_refresh` behavior can be added if explicit refresh is needed.
- Failed fetch diagnostics are stored as `last_error_json` and `error_count`.
- `error_count` counts failed external fetch attempts. Because `get_metadata()`
  uses retry/backoff, one logical call can increment this more than once.
- A later successful fetch clears `last_error_json` and resets `error_count` to
  `0`.

The cache table is diagnostic storage and lookup storage. It is not the circuit
breaker state.

## Provider Policy Contract

The provider clients and workflow worker share a provider-neutral failure
taxonomy. arXiv and Semantic Scholar also share Postgres-backed limiter and
circuit-breaker state in factory-created REST/worker processes.

Implementation status:

- DR.1 shared provider taxonomy is implemented.
- DR.2 Postgres-backed provider rate limiter is implemented for arXiv and
  Semantic Scholar factory-created REST/worker paths.
- DR.3 Postgres-backed provider circuit breaker is implemented for arXiv and
  Semantic Scholar factory-created REST/worker paths.
- DR.4 workflow retry scheduling uses taxonomy and provider delay hints.
- DR.5 health, diagnostics, and runbook coverage are implemented.

Distributed breaker coverage for LLM and embedding providers, authentication,
and production deployment hardening are outside this block.

Failure classes are intentionally provider-neutral:

| `failure_class` | Meaning |
| --- | --- |
| `rate_limited` | Provider rejected the request because rate or quota was exceeded. |
| `provider_unavailable` | External provider returned 5xx or connection failed. |
| `provider_timeout` | External provider timed out. |
| `provider_not_found` | Requested external paper/blob/resource does not exist. |
| `invalid_input` | Local validation or deterministic bad input. |
| `dependency_unavailable` | Required infrastructure dependency is unavailable. |
| `canceled` | User or worker cancellation interrupted work. |
| `internal_error` | Unexpected failure after known classes are exhausted. |

Initial provider policy:

| Provider | Operation | Retryable errors | Breaker failure? | Degradation allowed? | User-visible `failure_class` |
| --- | --- | --- | --- | --- | --- |
| arXiv | metadata lookup | 429, HTTP 5xx, timeout, transport error | Yes for HTTP 5xx, timeout, transport error; 429 feeds limiter policy | Yes, if PDF text is available and fallback metadata can be produced | `rate_limited`, `provider_unavailable`, `provider_timeout`, `provider_not_found` |
| arXiv | search | 429, HTTP 5xx, timeout, transport error | Yes for HTTP 5xx, timeout, transport error; 429 feeds limiter policy | Partial query failure may return warnings; total provider failure must remain visible as no/failed discovery rather than fake success | `rate_limited`, `provider_unavailable`, `provider_timeout` |
| Semantic Scholar | paper enrichment | 429, HTTP 5xx, timeout, transport error | Yes for HTTP 5xx, timeout, transport error; 429 feeds limiter policy | Yes, continue without citation enrichment | `rate_limited`, `provider_unavailable`, `provider_timeout`, `provider_not_found` |
| Semantic Scholar | recommendations/related papers | 429, HTTP 5xx, timeout, transport error | Yes for HTTP 5xx, timeout, transport error; 429 feeds limiter policy | Yes, continue with fewer/no related papers and warnings where surfaced | `rate_limited`, `provider_unavailable`, `provider_timeout`, `provider_not_found` |
| OpenAI embeddings | chunk/query embeddings | 429, HTTP 5xx, timeout, transport error | Future; not in first distributed-resilience slice | No. Retrieval indexing/search cannot proceed without embeddings | `rate_limited`, `provider_unavailable`, `provider_timeout` |
| Qdrant | vector collection/upsert/search | dependency/transport failure | No external-provider breaker in first slice | No. Analysis/retrieval must fail explicitly or retry as infrastructure failure | `dependency_unavailable` |
| Blob store | PDF object put/get/delete/materialize | dependency/transport failure | No external-provider breaker in first slice | No for required PDF materialization; cleanup may retry later | `dependency_unavailable`, `provider_not_found` for missing object |
| Workflow worker | job execution wrapper | shared taxonomy marks retryable provider/infra failures | N/A | N/A; retryable failures return to `queued` while attempts remain | same `failure_class` as source error |

The shared taxonomy in `services.provider_policy` is used by provider clients
and worker retry decisions. Factory-created REST/worker processes use
Postgres-backed limiter and circuit-breaker state for arXiv and Semantic
Scholar.

## Rate Limiting

arXiv and Semantic Scholar support a Postgres-backed distributed limiter for
REST and worker processes created by the application factory:

- arXiv: one request every `3.2` seconds.
- Semantic Scholar: one request every `1.2` seconds.

Limiter rows are keyed by provider and operation in `provider_rate_limits`.
arXiv metadata and discovery search share the `arxiv/api` key because both use
the same upstream request budget. Semantic Scholar enrichment uses
`semantic_scholar/api`. Reservation uses a transaction and row lock so
concurrent processes receive distinct future slots. `429` responses remain
retryable `rate_limited` events and do not count as breaker failures.

If the Postgres limiter is unavailable, the runtime limiter fails open with a
warning and applies a local interval sleep so analysis is not blocked by
coordination failure. Direct module usage without factory wiring falls back to
process-local limiting.

## Circuit Breakers

PaperIntel supports Postgres-backed circuit breakers for external API health in
REST and worker processes created by the application factory. Breaker rows are
keyed by provider and operation in `provider_circuit_breakers`.

Current parameters:

| Service | Opens after | Half-open after | Counted as failures | Not counted as failures |
| --- | ---: | ---: | --- | --- |
| arXiv | 5 consecutive external failures | 120 seconds | HTTP 5xx, timeouts, connection errors | 429, paper-not-found/404, local validation errors such as PDF too large |
| Semantic Scholar | 3 consecutive external failures | 60 seconds | HTTP 5xx, timeouts, connection errors | 403, 404, 429 |

arXiv metadata, download, and discovery search share the `arxiv/api` breaker
key. Semantic Scholar enrichment uses `semantic_scholar/api`. When a breaker is
open, calls fail fast internally with `CircuitBreakerOpenError` instead of
repeatedly calling a known-unhealthy upstream service. Public/job output must
surface this as `failure_class=provider_unavailable` with neutral text, not
internal breaker wording.

After the recovery timeout, one shared half-open probe is allowed. Other
processes continue to receive open decisions until the probe records success or
failure. A reachable non-fatal response closes the breaker; another external
failure opens it again. Direct module usage without factory wiring falls back to
the in-memory process-local breaker.

## Graceful Degradation

arXiv metadata failure alone should not fail paper analysis. The URL ingestion
path now follows this fallback order:

```text
1. Try arXiv metadata cache.
2. If cache misses, try arXiv metadata API subject to limiter and breaker.
3. Try PDF download and parsing.
4. If metadata is unavailable but PDF text is available, continue with
   pdf_fallback metadata.
5. If both metadata and PDF/text are unavailable, fail analysis.
```

Fallback metadata contains:

- `arxiv_id` from the URL.
- title from PDF metadata when available.
- `published_date` derived from the arXiv ID month, for example
  `2501.12948 -> 2025-01`.
- empty authors/categories when unavailable.
- Semantic Scholar citation count if enrichment succeeded.

Semantic Scholar enrichment is non-fatal. If S2 fails, rate-limits, or the
breaker is open, ingestion continues without citation enrichment.

## Worker Retry Semantics

Workflow job retry decisions use the shared provider taxonomy. Retryable classes
are `rate_limited`, `provider_unavailable`, `provider_timeout`, and
`dependency_unavailable`. `provider_not_found`, `invalid_input`, `canceled`, and
`internal_error` fail fast by default.

Job error JSON includes `failure_class` and `retryable`. The `retryable` field
means the current job will actually be retried; it is false on terminal failed
jobs after the retry budget is exhausted. When the provider gives a delay hint,
the worker also records `retry_after_seconds` for retried attempts and passes it
to the Postgres workflow job repository. The repository schedules the next
attempt using the larger of the normal exponential backoff and the provider
delay hint. This prevents fast retry loops when a provider is rate-limited or a
shared breaker is open.

Open circuit breakers are stored as retryable workflow failures while attempts
remain. Public/job messages stay neutral, for example
`Provider is temporarily unavailable`, and must not expose internal breaker
wording.

## Operational Notes

- Process-local limiters and breakers are direct-module fallback modes. Factory
  created REST/worker processes use Postgres coordination for arXiv and
  Semantic Scholar.
- `/health` reports `provider_resilience_store=ok` when Postgres-backed limiter
  and breaker tables are reachable. Open provider breakers do not make the app
  health check fail.
- For operational SQL and response guidance, see
  `docs/DISTRIBUTED_RESILIENCE_RUNBOOK.md`.
- The workflow worker should be run conservatively when many jobs use arXiv URLs.
- Worker retry decisions use the shared taxonomy in `services.provider_policy`
  plus provider delay hints such as `Retry-After`.
- For reproducible evaluation, prefer local PDFs plus golden metadata fallback
  where available.
- The async jobs live smoke intentionally uses a real arXiv URL and therefore can
  exercise the resilience layer, but it is still a happy-path smoke rather than
  a forced-outage test.

## Future Hardening

Deferred work:

- explicit `force_refresh` for arXiv metadata cache entries;
- richer provider observability and standardized trace labels;
- distributed resilience coverage for LLM and embedding providers after
  arXiv/Semantic Scholar policy is stable.
