import logging
import re
from typing import Optional

from models.schemas import PaperMetadata
from models.state import IngestionProvenance, PaperIntelState
from tools.arxiv_client import download_pdf, get_metadata
from tools.pdf_parser import parse_pdf
from tools.semantic_scholar_client import get_paper as s2_get_paper

logger = logging.getLogger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# Only the new arXiv ID format (2001.00001) is supported.
# Legacy ids like cs.AI/0601001 are a conscious limitation.
_ARXIV_ID_PATTERNS = [
    r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
    r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?",
    r"ar[Xx]iv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)",
]


def _sanitize_text(text: str) -> str:
    """Remove characters that PostgreSQL JSON/text cannot store safely."""
    if not text:
        return text
    text = text.replace("\x00", "")
    text = _SURROGATE_RE.sub("", text)
    return text


def _sanitize_text_by_page(text_by_page: dict[int, str]) -> dict[int, str]:
    return {page: _sanitize_text(text) for page, text in text_by_page.items()}


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _extract_arxiv_id(text: str) -> Optional[str]:
    for pattern in _ARXIV_ID_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return _strip_version(match.group(1))
    return None


def _enrich_s2(arxiv_id: str) -> Optional[dict]:
    """Non-blocking enrichment that never raises to the caller."""
    try:
        result = s2_get_paper(arxiv_id)
        logger.info(
            "S2 enrichment ok for %s: citations=%s",
            arxiv_id,
            result.get("citation_count"),
        )
        return result
    except Exception as exc:
        logger.warning("S2 enrichment failed for %s: %s", arxiv_id, exc)
        return None


def _resolve_metadata(
    arxiv_id: str,
    s2_data: Optional[dict],
) -> tuple[Optional[PaperMetadata], Optional[str]]:
    """
    Return (metadata, error_reason).
    error_reason is populated only when metadata resolution fails.
    """
    try:
        arxiv_meta = get_metadata(arxiv_id)
    except Exception as exc:
        reason = f"arXiv metadata failed for {arxiv_id}: {exc}"
        logger.exception("arXiv metadata error for %s", arxiv_id)
        return None, reason

    citation_count = s2_data.get("citation_count") if s2_data else None

    metadata = PaperMetadata(
        title=arxiv_meta.title,
        authors=arxiv_meta.authors,
        arxiv_id=arxiv_meta.arxiv_id,
        published_date=arxiv_meta.published_date,
        abstract=arxiv_meta.abstract,
        categories=arxiv_meta.categories,
        citation_count=citation_count,
    )
    return metadata, None


def _make_provenance(
    text_source: str,
    metadata_source: str,
    enrichment_status: str,
    arxiv_id_found: bool,
) -> IngestionProvenance:
    return {
        "text_source": text_source,
        "metadata_source": metadata_source,
        "enrichment_status": enrichment_status,
        "arxiv_id_found": arxiv_id_found,
    }


def _success_extraction(state: PaperIntelState, **kwargs) -> dict:
    """Successful ingestion that advances the graph to extraction."""
    return {
        "processing_stage": "extraction",
        **kwargs,
    }


def _failure(state: PaperIntelState, reason: str, level: str = "error") -> dict:
    if level == "error":
        logger.error("Ingestion failed: %s", reason)
    else:
        logger.warning("Ingestion warning: %s", reason)
    return {
        "errors": [reason],
        "processing_stage": "failed",
    }


def _validate_input(state: PaperIntelState) -> Optional[str]:
    input_type = state.get("input_type", "")
    input_value = state.get("input_value", "")

    if not input_type:
        return "input_type is missing"
    if not input_value or not input_value.strip():
        return "input_value is empty"
    if input_type not in ("url", "pdf", "topic_query"):
        return f"Unknown input_type: {input_type!r}"
    if input_type == "url" and not re.search(r"https?://", input_value):
        return f"input_value does not look like a URL: {input_value!r}"

    return None


def _route_url(state: PaperIntelState) -> dict:
    url = state["input_value"]
    arxiv_id = _extract_arxiv_id(url)

    if not arxiv_id:
        return _failure(state, f"Cannot extract arXiv ID from URL: {url}")

    logger.info("Ingestion [url] arxiv_id=%s", arxiv_id)

    s2_data = _enrich_s2(arxiv_id)
    enrichment = "s2_ok" if s2_data else "s2_failed"

    metadata, meta_error = _resolve_metadata(arxiv_id, s2_data)
    if metadata is None:
        return _failure(state, meta_error or f"Metadata unavailable for {arxiv_id}")

    try:
        pdf_path = download_pdf(arxiv_id)
        parsed = parse_pdf(pdf_path)
        raw_text = _sanitize_text(parsed["raw_text"])

        if not raw_text or not raw_text.strip():
            raise ValueError("PDF parsed but raw_text is empty")

        return _success_extraction(
            state,
            metadata=metadata,
            raw_text=raw_text,
            pdf_path=pdf_path,
            text_by_page=_sanitize_text_by_page(parsed["text_by_page"]),
            ingestion_provenance=_make_provenance(
                text_source="pdf",
                metadata_source="arxiv",
                enrichment_status=enrichment,
                arxiv_id_found=True,
            ),
        )
    except Exception as exc:
        logger.warning("PDF failed for %s, using abstract fallback: %s", arxiv_id, exc)
        return _success_extraction(
            state,
            metadata=metadata,
            raw_text=_sanitize_text(metadata.abstract),
            pdf_path=pdf_path if "pdf_path" in locals() else None,
            text_by_page=None,
            errors=[f"PDF unavailable, abstract used: {exc}"],
            ingestion_provenance=_make_provenance(
                text_source="abstract_fallback",
                metadata_source="arxiv",
                enrichment_status=enrichment,
                arxiv_id_found=True,
            ),
        )


def _route_pdf(state: PaperIntelState) -> dict:
    pdf_path = state["input_value"]
    logger.info("Ingestion [pdf] path=%s", pdf_path)

    try:
        parsed = parse_pdf(pdf_path)
    except Exception as exc:
        return _failure(state, f"PDF parse failed: {exc}")

    raw_text = _sanitize_text(parsed["raw_text"])
    if not raw_text or not raw_text.strip():
        return _failure(state, "PDF parsed but raw_text is empty")

    arxiv_id = parsed.get("arxiv_id")

    if arxiv_id:
        logger.info("Found arXiv ID in PDF: %s", arxiv_id)
        s2_data = _enrich_s2(arxiv_id)
        enrichment = "s2_ok" if s2_data else "s2_failed"
        metadata, meta_error = _resolve_metadata(arxiv_id, s2_data)

        if metadata:
            return _success_extraction(
                state,
                metadata=metadata,
                raw_text=raw_text,
                pdf_path=pdf_path,
                text_by_page=_sanitize_text_by_page(parsed["text_by_page"]),
                ingestion_provenance=_make_provenance(
                    text_source="pdf",
                    metadata_source="arxiv",
                    enrichment_status=enrichment,
                    arxiv_id_found=True,
                ),
            )

        logger.warning("arXiv metadata failed for %s: %s", arxiv_id, meta_error)
        return _success_extraction(
            state,
            metadata=PaperMetadata(
                title=parsed.get("metadata", {}).get("title") or "",
                authors=[],
                arxiv_id=arxiv_id,
                published_date="",
                abstract="",
                categories=[],
                citation_count=None,
            ),
            raw_text=raw_text,
            pdf_path=pdf_path,
            text_by_page=_sanitize_text_by_page(parsed["text_by_page"]),
            errors=[f"arXiv metadata unavailable, PDF fallback used: {meta_error}"],
            ingestion_provenance=_make_provenance(
                text_source="pdf",
                metadata_source="pdf_fallback",
                enrichment_status=enrichment,
                arxiv_id_found=True,
            ),
        )

    pdf_meta = parsed.get("metadata", {})
    metadata = PaperMetadata(
        title=pdf_meta.get("title") or "",
        authors=[],
        arxiv_id=None,
        published_date="",
        abstract="",
        categories=[],
        citation_count=None,
    )

    return _success_extraction(
        state,
        metadata=metadata,
        raw_text=raw_text,
        pdf_path=pdf_path,
        text_by_page=_sanitize_text_by_page(parsed["text_by_page"]),
        errors=["No arXiv ID - metadata quality low"],
        ingestion_provenance=_make_provenance(
            text_source="pdf",
            metadata_source="pdf_fallback",
            enrichment_status="not_attempted",
            arxiv_id_found=False,
        ),
    )


def _route_topic_query(state: PaperIntelState) -> dict:
    """
    Honest no-op until supervisor/topic selection is implemented.
    Search results are not stored in messages or papers.
    """
    query = state["input_value"]
    logger.info(
        "Ingestion [topic_query] - supervisor not implemented yet, query=%s",
        query,
    )
    return _failure(
        state,
        "topic_query route requires supervisor implementation (Week 1 Day 5-7)",
        level="warning",
    )


def ingestion_agent(state: PaperIntelState) -> dict:
    """
    Ingestion Agent LangGraph node.
    Routes by input_type: url | pdf | topic_query.
    """
    validation_error = _validate_input(state)
    if validation_error:
        return _failure(state, validation_error, level="warning")

    input_type = state["input_type"]
    logger.info("Ingestion agent started: input_type=%s", input_type)

    if input_type == "url":
        return _route_url(state)
    elif input_type == "pdf":
        return _route_pdf(state)
    elif input_type == "topic_query":
        return _route_topic_query(state)
    else:
        return _failure(state, f"Unknown input_type: {input_type!r}", level="warning")
