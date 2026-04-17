import logging
import time
import httpx
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper"
S2_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
RATE_LIMIT_DELAY = 1.0

_timeout = httpx.Timeout(30.0, connect=5.0)
_client = httpx.Client(timeout=_timeout, follow_redirects=True)


def _rate_limit():
    time.sleep(RATE_LIMIT_DELAY)


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_paper(arxiv_id: str) -> dict:
    logger.info(f"S2 get_paper: {arxiv_id}")
    url = f"{S2_API_URL}/arXiv:{arxiv_id}"
    params = {
        "fields": "title,citationCount,influentialCitationCount,openAccessPdf,externalIds"
    }

    t0 = time.perf_counter()
    response = _client.get(url, params=params)
    response.raise_for_status()
    latency = time.perf_counter() - t0
    logger.info(f"S2 get_paper latency: {latency:.2f}s")

    data = response.json()
    _check_for_error(data, arxiv_id)

    _rate_limit()
    return _parse_paper(data)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_related_papers(arxiv_id: str, limit: int = 5) -> List[dict]:
    logger.info(f"S2 related papers: {arxiv_id}")
    params = {
        "paperId": f"arXiv:{arxiv_id}",
        "fields": "title,year,citationCount,externalIds,score",
        "limit": limit,
    }

    t0 = time.perf_counter()
    response = _client.get(S2_RECOMMENDATIONS_URL, params=params)
    latency = time.perf_counter() - t0
    logger.info(f"S2 related latency: {latency:.2f}s")

    if response.status_code == 404:
        logger.warning(f"S2 related not found for {arxiv_id}")
        return []

    response.raise_for_status()
    data = response.json()
    _check_for_error(data, arxiv_id)

    papers = data.get("recommendedPapers", [])
    _rate_limit()
    return [_parse_related(p) for p in papers]