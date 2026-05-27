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
- For reproducible evaluation, prefer local PDFs plus golden metadata fallback
  where available.
- The async jobs live smoke intentionally uses a real arXiv URL and therefore can
  exercise the resilience layer, but it is still a happy-path smoke rather than
  a forced-outage test.

## Future Hardening

Deferred work:

- distributed rate limiter for multiple REST/worker processes;
- persistent or shared circuit breaker state if multi-process coordination is
  needed;
- explicit `force_refresh` for arXiv metadata cache entries;
- richer provider observability and standardized trace labels;
- separate PDF/local-file async job kinds.
