import logging
import time
from typing import Literal, Optional, TypedDict

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api"
RATE_LIMIT_DELAY = 0.5

_timeout = httpx.Timeout(30.0, connect=5.0)
_client = httpx.Client(timeout=_timeout, follow_redirects=True)

Reason = Literal["paper_not_found", "source_unavailable", None]


class PaperResource(TypedDict):
    repo_id: Optional[str]
    repo_type: Optional[Literal["model", "dataset", "space"]]
    url: Optional[str]
    likes: int
    downloads: Optional[int]
    paper_title: Optional[str]


class PaperResourcesResponse(TypedDict):
    source: Literal["huggingface"]
    source_available: bool
    paper_found: Optional[bool]
    reason: Reason
    results: list[PaperResource]


class PaperLookupResponse(TypedDict):
    source: Literal["huggingface"]
    source_available: bool
    paper_found: Optional[bool]
    reason: Reason
    title: Optional[str]
    summary: Optional[str]
    authors: list[str]


def _rate_limit() -> None:
    time.sleep(RATE_LIMIT_DELAY)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


_retry = retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


def _raise_for_status(response: httpx.Response, context: str) -> None:
    status = response.status_code
    if status == 404:
        return
    if status == 429:
        logger.warning("HF rate limit hit: %s", context)
        response.raise_for_status()
    if status >= 500:
        logger.error("HF server error %s: %s", status, context)
        response.raise_for_status()
    if status >= 400:
        logger.error("HF client error %s: %s", status, context)
        response.raise_for_status()


@_retry
def _request(path: str, *, context: str) -> httpx.Response:
    t0 = time.perf_counter()
    response = _client.get(f"{HF_API_URL}{path}")
    latency = time.perf_counter() - t0
    logger.info("HF %s latency: %.2fs", context, latency)
    _raise_for_status(response, context)
    _rate_limit()  # только если запрос успешен, не в finally
    return response


def _safe_json(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        logger.warning("HF returned invalid JSON")
        return {}
    if not isinstance(payload, dict):
        logger.warning("HF returned non-dict payload")
        return {}
    return payload


def _build_repo_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def _parse_resource(item: dict) -> PaperResource:
    repo_id = item.get("id")
    repo_type = item.get("type")

    normalized_type: Optional[Literal["model", "dataset", "space"]]
    if repo_type in {"model", "dataset", "space"}:
        normalized_type = repo_type
    else:
        normalized_type = None

    url = None
    if repo_id and normalized_type:
        url = _build_repo_url(repo_id, normalized_type)

    paper = item.get("paper")
    paper_title = paper.get("title") if isinstance(paper, dict) else None

    downloads = item.get("downloads")
    if downloads is not None:
        try:
            downloads = int(downloads)
        except (TypeError, ValueError):
            downloads = None

    return {
        "repo_id": repo_id,
        "repo_type": normalized_type,
        "url": url,
        "likes": int(item.get("likes") or 0),
        "downloads": downloads,
        "paper_title": paper_title,
    }


def get_paper(arxiv_id: str) -> PaperLookupResponse:
    logger.info("HF paper lookup: %s", arxiv_id)
    try:
        response = _request(
            f"/papers/{arxiv_id}",
            context=f"paper lookup {arxiv_id}",
        )
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
        logger.exception("HF paper lookup failed for %s", arxiv_id)
        return {
            "source": "huggingface",
            "source_available": False,
            "paper_found": None,
            "reason": "source_unavailable",
            "title": None,
            "summary": None,
            "authors": [],
        }

    if response.status_code == 404:
        return {
            "source": "huggingface",
            "source_available": True,
            "paper_found": False,
            "reason": "paper_not_found",
            "title": None,
            "summary": None,
            "authors": [],
        }

    payload = _safe_json(response)
    authors_raw = payload.get("authors", [])
    authors: list[str] = []

    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, dict):
                name = author.get("name")
                if isinstance(name, str) and name.strip():
                    authors.append(name.strip())
            elif isinstance(author, str) and author.strip():
                authors.append(author.strip())

    return {
        "source": "huggingface",
        "source_available": True,
        "paper_found": True,
        "reason": None,
        "title": payload.get("title"),
        "summary": payload.get("summary"),
        "authors": authors,
    }


def get_resources(arxiv_id: str) -> PaperResourcesResponse:
    logger.info("HF resources lookup: %s", arxiv_id)
    try:
        response = _request(
            f"/arxiv/{arxiv_id}/repos",
            context=f"repos {arxiv_id}",
        )
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
        logger.exception("HF resources lookup failed for %s", arxiv_id)
        return {
            "source": "huggingface",
            "source_available": False,
            "paper_found": None,
            "reason": "source_unavailable",
            "results": [],
        }

    if response.status_code == 404:
        return {
            "source": "huggingface",
            "source_available": True,
            "paper_found": False,
            "reason": "paper_not_found",
            "results": [],
        }

    payload = _safe_json(response)
    raw_results = payload.get("repos", payload.get("results", []))

    if not isinstance(raw_results, list):
        logger.warning("HF repos payload has non-list results for %s", arxiv_id)
        raw_results = []

    return {
        "source": "huggingface",
        "source_available": True,
        "paper_found": True,
        "reason": None,
        "results": [_parse_resource(item) for item in raw_results if isinstance(item, dict)],
    }