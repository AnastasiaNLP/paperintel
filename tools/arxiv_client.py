import logging
import time
import httpx
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf"
NS = "{http://www.w3.org/2005/Atom}"
MAX_PDF_SIZE_MB = 50
RATE_LIMIT_DELAY = 0.4  # seconds between requests

#  one client for the entire module that maintains a keep-alive connection
_client = httpx.Client(timeout=30, follow_redirects=True)


def _rate_limit():
    time.sleep(RATE_LIMIT_DELAY)


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_papers(query: str, max_results: int = 10) -> List[dict]:
    logger.info(f"arXiv search: '{query}' max={max_results}")
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    t0 = time.perf_counter()
    response = _client.get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    latency = time.perf_counter() - t0
    logger.info(f"arXiv search latency: {latency:.2f}s")

    root = ET.fromstring(response.text)
    papers = [_parse_entry(e) for e in root.findall(f"{NS}entry")]
    logger.info(f"Found {len(papers)} papers")

    _rate_limit()
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_metadata(arxiv_id: str) -> PaperMetadata:
    logger.info(f"arXiv metadata: {arxiv_id}")
    params = {"id_list": arxiv_id}

    t0 = time.perf_counter()
    response = _client.get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    latency = time.perf_counter() - t0
    logger.info(f"arXiv metadata latency: {latency:.2f}s")

    root = ET.fromstring(response.text)
    entry = root.find(f"{NS}entry")
    if entry is None:
        raise ValueError(f"Paper not found: {arxiv_id}")

    data = _parse_entry(entry)
    _rate_limit()

    return PaperMetadata(
        title=data["title"],
        authors=data["authors"],
        arxiv_id=arxiv_id,
        published_date=data["published_date"],
        abstract=data["abstract"],
        categories=data["categories"],
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_pdf(arxiv_id: str, save_dir: str = "tmp") -> str:
    Path(save_dir).mkdir(exist_ok=True)
    pdf_path = Path(save_dir) / f"{arxiv_id.replace('/', '_')}.pdf"

    if pdf_path.exists():
        logger.info(f"PDF cached: {pdf_path}")
        return str(pdf_path)

    url = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"
    logger.info(f"Downloading PDF: {url}")

    t0 = time.perf_counter()
    with _client.stream("GET", url, follow_redirects=True) as response:
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

    _rate_limit()
    return str(pdf_path)