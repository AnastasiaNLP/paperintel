import json
import logging
import re
from pathlib import Path
from typing import Optional

from agents.error_utils import paper_error
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.schemas import MethodExtraction
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "extraction_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

HEAD_CHARS = 60_000
TAIL_CHARS = 20_000
CONFIDENCE_THRESHOLD = 0.7

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _select_relevant_text(raw_text: str) -> str:
    """
    Select paper sections likely to contain method, baselines, and limitations.
    Strategy: beginning + tail rather than naive head-only truncation.
    """
    if len(raw_text) <= HEAD_CHARS + TAIL_CHARS:
        return raw_text

    head = raw_text[:HEAD_CHARS]
    tail = raw_text[-TAIL_CHARS:]
    selected = head + "\n\n[... middle sections omitted ...]\n\n" + tail
    logger.info(
        "Text selected: head=%d + tail=%d = %d chars (original=%d)",
        len(head),
        len(tail),
        len(selected),
        len(raw_text),
    )
    return selected


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` if the model returns fenced JSON."""
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_confidence(raw: object) -> float:
    """Safely parse confidence and clamp it to [0.0, 1.0]."""
    try:
        value = float(raw)  # type: ignore[arg-type]
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))
    except (TypeError, ValueError):
        logger.warning("Could not parse confidence value: %r; defaulting to 0.0", raw)
        return 0.0


def _as_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_as_string(item) for item in value) if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_extraction(raw_json: str) -> tuple[Optional[MethodExtraction], float, Optional[str]]:
    """
    Parse LLM JSON into (MethodExtraction, confidence, error_reason).
    """
    cleaned = _strip_json_fences(raw_json)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, 0.0, f"JSON parse error: {exc} | raw: {cleaned[:200]}"

    if not isinstance(data, dict):
        return None, 0.0, "JSON root must be an object"

    confidence = _parse_confidence(data.get("confidence", 0.0))

    try:
        extraction = MethodExtraction(
            method_name=_as_string(data.get("method_name")),
            description=_as_string(data.get("description")),
            novelty_claim=_as_string(data.get("novelty_claim")),
            key_components=_as_string_list(data.get("key_components")),
            compared_to=_as_string_list(data.get("compared_to")),
            limitations_stated=_as_string_list(data.get("limitations_stated")),
        )
        return extraction, confidence, None
    except Exception as exc:
        return None, 0.0, f"MethodExtraction validation error: {exc}"


def _build_user_message(text: str, metadata_header: Optional[str] = None) -> str:
    header = f"{metadata_header}\n\n" if metadata_header else ""
    return f"{header}Extract structured information from this research paper:\n\n{text}"


def _extract_text_block(response: object, *, context: str) -> tuple[Optional[str], Optional[str]]:
    content = getattr(response, "content", None)
    if not content:
        return None, f"{context} returned empty content"

    block = content[0]
    raw = getattr(block, "text", None)
    if not isinstance(raw, str) or not raw.strip():
        return None, f"{context} returned non-text or empty block: {type(block)}"

    return raw.strip(), None


def _call_llm(
    text: str,
    metadata_header: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Call Claude Sonnet and return (raw_json, error_reason).
    """
    return call_text_llm(
        requested_model=settings.haiku_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=_build_user_message(text, metadata_header),
        max_tokens=2000,
        context_label="LLM extraction",
    )


def _call_llm_repair(bad_json: str) -> tuple[Optional[str], Optional[str]]:
    """
    Repair-oriented retry: ask the model to convert the previous output to valid JSON.
    """
    return call_text_llm(
        requested_model=settings.haiku_model,
        system_prompt="You are a JSON repair specialist. Return ONLY valid JSON, no explanation.",
        user_content=(
            "The following JSON is invalid. Fix it and return ONLY the corrected JSON:\n\n"
            f"{bad_json[:4000]}"
        ),
        max_tokens=2000,
        context_label="Repair LLM",
    )


def _metadata_header(state: PaperIntelState, text_source: str) -> Optional[str]:
    metadata = state.get("metadata")
    if not metadata:
        return None

    return (
        f"Paper title: {metadata.title}\n"
        f"Abstract: {metadata.abstract[:500]}\n"
        f"Text source: {text_source}"
    )


def extraction_agent(state: PaperIntelState) -> dict:
    """
    Extraction Agent LangGraph node.
    Reads raw_text from state and extracts MethodExtraction via Claude Sonnet.
    """
    raw_text = state.get("raw_text")
    if not raw_text or not raw_text.strip():
        logger.error("Extraction agent: raw_text is empty")
        return paper_error(state, "Extraction: raw_text is empty", "extraction")

    provenance = state.get("ingestion_provenance") or {}
    text_source = provenance.get("text_source", "pdf")
    abstract_only = text_source == "abstract_fallback"
    max_confidence = 0.5 if abstract_only else 1.0

    if abstract_only:
        logger.warning("Text source is abstract_fallback; capping confidence at 0.5")

    logger.info(
        "Extraction agent started, text_length=%d abstract_only=%s",
        len(raw_text),
        abstract_only,
    )

    text = _select_relevant_text(raw_text)
    raw_json, llm_error = _call_llm(text, _metadata_header(state, text_source))

    if llm_error:
        logger.error("LLM call failed: %s", llm_error)
        return paper_error(state, llm_error, "extraction")

    extraction, confidence, parse_error = _parse_extraction(raw_json or "")

    if parse_error:
        logger.warning("Parse error: %s; attempting repair", parse_error)
        repaired_json, repair_error = _call_llm_repair(raw_json or "")

        if repair_error:
            logger.error("Repair failed: %s", repair_error)
            return paper_error(
                state,
                f"Extraction parse failed: {parse_error}; repair failed: {repair_error}",
                "extraction",
            )

        extraction, confidence, parse_error = _parse_extraction(repaired_json or "")

    if parse_error or extraction is None:
        logger.error("Extraction failed after repair: %s", parse_error)
        return paper_error(state, parse_error or "Extraction failed", "extraction")

    confidence = min(confidence, max_confidence)
    needs_review = confidence < CONFIDENCE_THRESHOLD

    if needs_review:
        logger.warning(
            "Low confidence: %.2f < %.2f; flagging for human review",
            confidence,
            CONFIDENCE_THRESHOLD,
        )

    logger.info(
        "Extraction complete: method=%r confidence=%.2f needs_review=%s",
        extraction.method_name,
        confidence,
        needs_review,
    )

    return {
        "method_extraction": extraction,
        "needs_human_review": needs_review,
        "human_review_reason": f"Low confidence: {confidence:.2f}" if needs_review else None,
        "processing_stage": "benchmark",
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "extraction": confidence,
        },
    }
