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

The current code has several local classifiers: arXiv metadata/search,
Semantic Scholar enrichment, OpenAI embeddings, blob storage, Qdrant, and the
workflow worker retry wrapper. The distributed-resilience work should converge
those classifiers on the policy below before adding shared limiter or breaker
state.

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

DR.1 should turn this table into a small shared taxonomy/API used by provider
clients and worker retry decisions. DR.2 should add the Postgres-backed
distributed limiter for arXiv and Semantic Scholar. DR.3 should add shared
circuit-breaker state after taxonomy and limiter semantics are stable.

## Rate Limiting

Both clients use process-local lock/timestamp limiters before requests:

- arXiv: one request every `3.2` seconds.
- Semantic Scholar: one request every `1.2` seconds.

This is intentionally process-local. If you run multiple REST/worker processes,
each process has its own limiter and the combined host/IP traffic can still
exceed upstream limits. For early self-hosted deployments, run a single
arXiv-using worker process unless you add a distributed limiter.

A distributed Postgres-backed limiter is not implemented yet.

## Circuit Breakers

PaperIntel uses in-memory circuit breakers for external API health. Breaker
state is process-local and resets when the process restarts.

Current parameters:

| Service | Opens after | Half-open after | Counted as failures | Not counted as failures |
| --- | ---: | ---: | --- | --- |
| arXiv | 5 consecutive external failures | 120 seconds | HTTP 5xx, 429, timeouts, connection errors | paper-not-found/404, local validation errors such as PDF too large |
| Semantic Scholar | 3 consecutive external failures | 60 seconds | HTTP 5xx, timeouts, connection errors | 403, 404, 429 |

When a breaker is open, calls fail fast with `CircuitBreakerOpenError` instead
of repeatedly calling a known-unhealthy upstream service. After the recovery
timeout, one half-open probe is allowed. A reachable non-fatal response closes
the breaker; another external failure opens it again.

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

## Operational Notes

- Process-local limiters and breakers are appropriate for the current local and
  early-OSS deployment shape. They are not a distributed coordination layer.
- The workflow worker should be run conservatively when many jobs use arXiv URLs.
- Worker retry decisions currently use a local helper in `workers.workflow_worker`;
  DR.1 should replace that local list with the shared taxonomy above.
- For reproducible evaluation, prefer local PDFs plus golden metadata fallback
  where available.
- The async jobs live smoke intentionally uses a real arXiv URL and therefore can
  exercise the resilience layer, but it is still a happy-path smoke rather than
  a forced-outage test.

## Future Hardening

Deferred work:

- distributed rate limiter for multiple REST/worker processes;
- shared provider error taxonomy used by arXiv, Semantic Scholar, and workflow
  worker retry decisions;
- persistent or shared circuit breaker state if multi-process coordination is
  needed;
- explicit `force_refresh` for arXiv metadata cache entries;
- richer provider observability and standardized trace labels;
- distributed resilience coverage for LLM and embedding providers after
  arXiv/Semantic Scholar policy is stable.
