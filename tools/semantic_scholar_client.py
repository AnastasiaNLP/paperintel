import logging
import os
import time
import httpx
from threading import Lock
from typing import List
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper"
S2_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
S2_REQUEST_INTERVAL_SECONDS = 1.2
# Backward-compatible alias for tests and older imports.
RATE_LIMIT_DELAY = S2_REQUEST_INTERVAL_SECONDS
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

_timeout = httpx.Timeout(30.0, connect=5.0)
_client = httpx.Client(timeout=_timeout, follow_redirects=True)
_rate_limit_lock = Lock()
_last_request_at = 0.0
_s2_breaker = CircuitBreaker(
    service_name="semantic_scholar",
    failure_threshold=3,
    recovery_timeout_seconds=60.0,
)


def reset_circuit_breaker() -> None:
    _s2_breaker.reset()


def _record_s2_success() -> None:
    _s2_breaker.record_success()


def _record_s2_failure(exc: Exception) -> None:
    if isinstance(exc, CircuitBreakerOpenError):
        return
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {403, 404, 429}:
        return
    _s2_breaker.record_failure()


def _rate_limit():
    global _last_request_at
    with _rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _last_request_at
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        _last_request_at = time.monotonic()


def _get(url: str, *, params: dict | None = None) -> httpx.Response:
    _rate_limit()
    return _client.get(url, params=params, headers=_headers())


def _headers() -> dict[str, str]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, CircuitBreakerOpenError):
        return False
    if not isinstance(exc, httpx.HTTPStatusError):
        return True
    return exc.response.status_code in RETRYABLE_STATUS_CODES


def _handle_non_retryable_status(response: httpx.Response, arxiv_id: str) -> bool:
    if response.status_code == 404:
        logger.info("S2 paper not found for %s", arxiv_id)
        return True
    if response.status_code == 403:
        logger.warning("S2 access forbidden for %s; continuing without enrichment", arxiv_id)
        return True
    if response.status_code == 429:
        logger.warning("S2 rate-limited for %s; continuing without enrichment", arxiv_id)
        return True
    return False


def _check_for_error(data: dict, arxiv_id: str):
    """S2 sometimes returns 200 + error inside body"""
    if "error" in data:
        raise ValueError(f"S2 API error for {arxiv_id}: {data['error']}")
    if "code" in data and "error" in data["code"].lower():
        raise ValueError(f"S2 API error for {arxiv_id}: {data.get('message', data['code'])}")
    if "message" in data and "not found" in str(data["message"]).lower():
        raise ValueError(f"Paper not found in S2: {arxiv_id}")


def _parse_paper(data: dict) -> dict:
    """Normalization layer - how _parse_entry in arxiv_client"""
    return {
        "s2_paper_id": data.get("paperId"),
        "citation_count": int(data.get("citationCount") or 0),
        "influential_citation_count": int(data.get("influentialCitationCount") or 0),
        "open_access_pdf": (
            data.get("openAccessPdf", {}).get("url")
            if data.get("openAccessPdf") else None
        ),
    }


def _parse_related(p: dict) -> dict:
    """Normalization for related papers"""
    return {
        "s2_paper_id": p.get("paperId"),
        "title": p.get("title"),
        "year": int(p.get("year") or 0),
        "citation_count": int(p.get("citationCount") or 0),
        "score": p.get("score"),
        "arxiv_id": p.get("externalIds", {}).get("ArXiv"),
    }


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def get_paper(arxiv_id: str) -> dict:
    logger.info(f"S2 get_paper: {arxiv_id}")
    url = f"{S2_API_URL}/arXiv:{arxiv_id}"
    params = {
        "fields": "title,citationCount,influentialCitationCount,openAccessPdf,externalIds"
    }

    t0 = time.perf_counter()
    try:
        _s2_breaker.before_request()
        response = _get(url, params=params)
        if _handle_non_retryable_status(response, arxiv_id):
            _record_s2_success()
            return {}
        response.raise_for_status()
        latency = time.perf_counter() - t0
        logger.info(f"S2 get_paper latency: {latency:.2f}s")

        data = response.json()
        _check_for_error(data, arxiv_id)
    except Exception as exc:
        _record_s2_failure(exc)
        raise

    _record_s2_success()
    return _parse_paper(data)


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def get_related_papers(arxiv_id: str, limit: int = 5) -> List[dict]:
    logger.info(f"S2 related papers: {arxiv_id}")
    params = {
        "paperId": f"arXiv:{arxiv_id}",
        "fields": "title,year,citationCount,externalIds,score",
        "limit": limit,
    }

    t0 = time.perf_counter()
    try:
        _s2_breaker.before_request()
        response = _get(S2_RECOMMENDATIONS_URL, params=params)
        latency = time.perf_counter() - t0
        logger.info(f"S2 related latency: {latency:.2f}s")

        if _handle_non_retryable_status(response, arxiv_id):
            _record_s2_success()
            return []

        response.raise_for_status()
        data = response.json()
        _check_for_error(data, arxiv_id)
    except Exception as exc:
        _record_s2_failure(exc)
        raise

    _record_s2_success()
    papers = data.get("recommendedPapers", [])
    return [_parse_related(p) for p in papers]
