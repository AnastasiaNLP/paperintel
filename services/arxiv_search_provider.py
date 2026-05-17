import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from models.discovery import RawSearchResult, ResearchQuery


logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
MAX_ARXIV_RESULTS = 25
RATE_LIMIT_DELAY = 0.4


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _retry():
    return retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )


class ArxivSearchProvider:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_results_cap: int = MAX_ARXIV_RESULTS,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=5.0),
            follow_redirects=True,
        )
        self.max_results_cap = max_results_cap
        self.rate_limit_delay = rate_limit_delay

    def search(self, query: ResearchQuery) -> list[RawSearchResult]:
        normalized_query = normalize_query(query.query)
        max_results = min(query.max_results, self.max_results_cap)
        response = self._request(normalized_query, max_results=max_results)
        return parse_arxiv_feed(response.text)

    @_retry()
    def _request(self, query: str, *, max_results: int) -> httpx.Response:
        params = {
            "search_query": build_search_query(query),
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        t0 = time.perf_counter()
        response = self.client.get(ARXIV_API_URL, params=params)
        latency = time.perf_counter() - t0
        logger.info("arXiv search latency: %.2fs", latency)
        response.raise_for_status()
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        return response

    def close(self) -> None:
        self.client.close()


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def build_search_query(query: str) -> str:
    if re.match(r"^(all|ti|au|abs|co|jr|cat|rn|id):", query):
        return query
    return f"all:{query}"


def parse_arxiv_feed(xml_text: str) -> list[RawSearchResult]:
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        parsed = parse_arxiv_entry(entry)
        if parsed is not None:
            results.append(parsed)
    return results


def parse_arxiv_entry(entry: ET.Element) -> RawSearchResult | None:
    raw_id = _safe_text(entry, f"{ATOM_NS}id")
    title = _clean_text(_safe_text(entry, f"{ATOM_NS}title"))
    if not raw_id or not title:
        return None

    arxiv_id = normalize_arxiv_id(raw_id)
    if not arxiv_id:
        return None

    published_at = _parse_datetime(_safe_text(entry, f"{ATOM_NS}published"))
    abstract = _clean_text(_safe_text(entry, f"{ATOM_NS}summary")) or None
    authors = []
    for author in entry.findall(f"{ATOM_NS}author"):
        name = _safe_text(author, f"{ATOM_NS}name")
        if name:
            authors.append(name)

    return RawSearchResult(
        title=title,
        url=canonical_abs_url(arxiv_id),
        source="arxiv",
        authors=authors,
        year=published_at.year if published_at else None,
        arxiv_id=arxiv_id,
        abstract=abstract,
        published_at=published_at,
        metadata={"raw_id": raw_id},
    )


def normalize_arxiv_id(value: str) -> str | None:
    value = value.strip()
    if "/abs/" in value:
        value = value.rsplit("/abs/", 1)[-1]
    if "/pdf/" in value:
        value = value.rsplit("/pdf/", 1)[-1]
    value = value.removesuffix(".pdf")
    value = re.sub(r"v\d+$", "", value)
    return value or None


def canonical_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _safe_text(element: ET.Element, path: str) -> str | None:
    child = element.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
