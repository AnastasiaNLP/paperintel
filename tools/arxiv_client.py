import logging
import time
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional, Protocol
from tenacity import retry, stop_after_attempt, wait_exponential
from models.external_metadata import ArxivMetadataCacheEntry
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)


class MetadataCacheRepository(Protocol):
    def get(self, arxiv_id: str) -> ArxivMetadataCacheEntry | None:
        ...

    def record_success(
        self, entry: ArxivMetadataCacheEntry
    ) -> ArxivMetadataCacheEntry:
        ...

    def record_error(
        self, arxiv_id: str, *, error_json: dict
    ) -> ArxivMetadataCacheEntry:
        ...


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf"
NS = "{http://www.w3.org/2005/Atom}"
MAX_PDF_SIZE_MB = 50
# arXiv API ToU asks legacy API clients to make no more than one request every
# three seconds and use one connection at a time.
ARXIV_REQUEST_INTERVAL_SECONDS = 3.2
# Backward-compatible alias for tests and older imports.
RATE_LIMIT_DELAY = ARXIV_REQUEST_INTERVAL_SECONDS

#  one client for the entire module that maintains a keep-alive connection
_client = httpx.Client(timeout=30, follow_redirects=True)
_rate_limit_lock = Lock()
_last_request_at = 0.0
_metadata_cache_repository: MetadataCacheRepository | None = None


def configure_metadata_cache(repository: MetadataCacheRepository | None) -> None:
    global _metadata_cache_repository
    _metadata_cache_repository = repository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _metadata_from_cache_entry(entry: ArxivMetadataCacheEntry) -> PaperMetadata:
    return PaperMetadata(
        title=entry.title or "",
        authors=entry.authors,
        arxiv_id=entry.arxiv_id,
        published_date=entry.published_date or "",
        abstract=entry.abstract or "",
        categories=entry.categories,
    )


def _cache_entry_from_metadata(metadata: PaperMetadata) -> ArxivMetadataCacheEntry:
    return ArxivMetadataCacheEntry(
        arxiv_id=metadata.arxiv_id or "",
        title=metadata.title,
        authors=metadata.authors,
        abstract=metadata.abstract,
        published_date=metadata.published_date,
        categories=metadata.categories,
        source_url=(
            f"https://arxiv.org/abs/{metadata.arxiv_id}"
            if metadata.arxiv_id
            else None
        ),
        fetched_at=_utc_now(),
    )


def _read_metadata_cache(arxiv_id: str) -> PaperMetadata | None:
    if _metadata_cache_repository is None:
        return None
    try:
        entry = _metadata_cache_repository.get(arxiv_id)
    except Exception as exc:
        logger.warning("arXiv metadata cache read failed for %s: %s", arxiv_id, exc)
        return None
    if entry is None or not entry.has_successful_fetch:
        return None
    logger.info("arXiv metadata cache hit: %s", arxiv_id)
    return _metadata_from_cache_entry(entry)


def _record_metadata_cache_success(metadata: PaperMetadata) -> None:
    if _metadata_cache_repository is None or not metadata.arxiv_id:
        return
    try:
        _metadata_cache_repository.record_success(_cache_entry_from_metadata(metadata))
    except Exception as exc:
        logger.warning(
            "arXiv metadata cache write failed for %s: %s",
            metadata.arxiv_id,
            exc,
        )


def _record_metadata_cache_error(arxiv_id: str, exc: Exception) -> None:
    if _metadata_cache_repository is None:
        return
    error_json = {
        "code": exc.__class__.__name__,
        "message": str(exc),
        "timestamp": _utc_now().isoformat(),
    }
    response = getattr(exc, "response", None)
    if response is not None:
        error_json["http_status"] = getattr(response, "status_code", None)
    try:
        _metadata_cache_repository.record_error(arxiv_id, error_json=error_json)
    except Exception as cache_exc:
        logger.warning(
            "arXiv metadata cache error write failed for %s: %s",
            arxiv_id,
            cache_exc,
        )


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
    return _client.get(url, params=params)


def _stream(method: str, url: str, **kwargs):
    _rate_limit()
    return _client.stream(method, url, **kwargs)


def _safe_text(element, path: str) -> Optional[str]:
    el = element.find(path)
    return el.text.strip() if el is not None and el.text else None


def _parse_entry(entry) -> dict:
    raw_id = _safe_text(entry, f"{NS}id") or ""
    arxiv_id = raw_id.split("/abs/")[-1]

    title = (_safe_text(entry, f"{NS}title") or "").replace("\n", " ")
    abstract = (_safe_text(entry, f"{NS}summary") or "").replace("\n", " ")
    published = (_safe_text(entry, f"{NS}published") or "")[:10]

    authors = [
        a.find(f"{NS}name").text
        for a in entry.findall(f"{NS}author")
        if a.find(f"{NS}name") is not None
    ]

    # all categories
    categories = [
        c.get("term")
        for c in entry.findall(f"{NS}category")
        if c.get("term")
    ]

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "published_date": published,
        "authors": authors,
        "categories": categories,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=3, min=10, max=60))
def search_papers(query: str, max_results: int = 10) -> List[dict]:
    logger.info(f"arXiv search: '{query}' max={max_results}")
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    t0 = time.perf_counter()
    response = _get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    latency = time.perf_counter() - t0
    logger.info(f"arXiv search latency: {latency:.2f}s")

    root = ET.fromstring(response.text)
    papers = [_parse_entry(e) for e in root.findall(f"{NS}entry")]
    logger.info(f"Found {len(papers)} papers")

    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=3, min=10, max=60))
def get_metadata(arxiv_id: str) -> PaperMetadata:
    cached = _read_metadata_cache(arxiv_id)
    if cached is not None:
        return cached

    logger.info(f"arXiv metadata: {arxiv_id}")
    params = {"id_list": arxiv_id}

    try:
        t0 = time.perf_counter()
        response = _get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        latency = time.perf_counter() - t0
        logger.info(f"arXiv metadata latency: {latency:.2f}s")

        root = ET.fromstring(response.text)
        entry = root.find(f"{NS}entry")
        if entry is None:
            raise ValueError(f"Paper not found: {arxiv_id}")

        data = _parse_entry(entry)
        metadata = PaperMetadata(
            title=data["title"],
            authors=data["authors"],
            arxiv_id=arxiv_id,
            published_date=data["published_date"],
            abstract=data["abstract"],
            categories=data["categories"],
        )
    except Exception as exc:
        _record_metadata_cache_error(arxiv_id, exc)
        raise

    _record_metadata_cache_success(metadata)
    return metadata


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=3, min=10, max=60))
def download_pdf(arxiv_id: str, save_dir: str = "tmp") -> str:
    Path(save_dir).mkdir(exist_ok=True)
    pdf_path = Path(save_dir) / f"{arxiv_id.replace('/', '_')}.pdf"

    if pdf_path.exists():
        logger.info(f"PDF cached: {pdf_path}")
        return str(pdf_path)

    url = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"
    logger.info(f"Downloading PDF: {url}")

    t0 = time.perf_counter()
    with _stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > MAX_PDF_SIZE_MB:
                raise ValueError(f"PDF too large: {size_mb:.1f}MB")

        downloaded = 0
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_PDF_SIZE_MB * 1024 * 1024:
                    pdf_path.unlink(missing_ok=True)
                    raise ValueError(f"PDF exceeded {MAX_PDF_SIZE_MB}MB")
                f.write(chunk)

    latency = time.perf_counter() - t0
    logger.info(f"Downloaded {downloaded / 1024 / 1024:.1f}MB in {latency:.2f}s → {pdf_path}")

    return str(pdf_path)
