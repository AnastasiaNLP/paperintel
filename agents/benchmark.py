import json
import logging
import re
from pathlib import Path
from typing import Optional

import anthropic

from agents.error_utils import paper_error
from config.settings import settings
from models.schemas import BenchmarkResult
from models.state import PaperIntelState
from tools.pdf_parser import extract_tables

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "benchmark_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

MAX_TABLE_CHARS = 14_000
MAX_PAGE_CONTEXT_CHARS = 1_500
MAX_FALLBACK_TEXT_CHARS = 12_000
MAX_BENCHMARK_CONTEXT_CHARS = 18_000
BENCHMARK_CONTEXT_WINDOW_CHARS = 1_200
MAX_BENCHMARK_CONTEXT_WINDOWS = 24
BENCHMARK_CONTEXT_KEYWORDS = [
    "MATH-500",
    "MATH",
    "AIME",
    "GPQA",
    "LiveCodeBench",
    "Codeforces",
    "HumanEval",
    "MBPP",
    "MMLU",
    "MMLU-Pro",
    "SWE-bench",
    "GSM8K",
    "DROP",
    "BBH",
    "IFEval",
    "AlpacaEval",
    "Arena-Hard",
    "pass@1",
    "cons@",
    "accuracy",
    "benchmark",
    "evaluation",
    "results",
    "Table",
]


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _as_optional_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(",", "")
    else:
        cleaned = value
    try:
        return float(cleaned)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _proposed_method_name(state: PaperIntelState) -> str:
    extraction = state.get("method_extraction")
    if extraction and extraction.method_name:
        return extraction.method_name

    metadata = state.get("metadata")
    if metadata and metadata.title:
        return metadata.title

    return "the proposed method"


def _format_rows(rows: list[list[object]]) -> str:
    formatted_rows = []
    for row in rows:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        formatted_rows.append(" | ".join(cells))
    return "\n".join(formatted_rows)


def _format_tables_with_context(
    tables: list,
    text_by_page: Optional[dict[int, str]],
) -> tuple[str, bool]:
    if not tables:
        return "No tables extracted from PDF.", False

    parts: list[str] = []

    for index, table in enumerate(tables, start=1):
        rows = table.get("rows", [])
        page = table.get("page")
        needs_vision = bool(table.get("needs_vision", False))

        header = f"Table {index}"
        if page is not None:
            header += f" (page {page})"
        if needs_vision:
            header += " [complex layout: extracted rows may be incomplete]"

        table_text = f"{header}:\n{_format_rows(rows)}"

        if text_by_page and page in text_by_page:
            page_context = text_by_page[page][:MAX_PAGE_CONTEXT_CHARS]
            table_text += f"\n\nPage {page} context:\n{page_context}"

        parts.append(table_text)

    result = "\n\n---\n\n".join(parts)
    truncated = len(result) > MAX_TABLE_CHARS
    if truncated:
        logger.info(
            "Tables context truncated from %d to %d chars",
            len(result),
            MAX_TABLE_CHARS,
        )
        result = result[:MAX_TABLE_CHARS] + "\n[truncated]"
    return result, truncated


def _format_fallback_text(raw_text: Optional[str]) -> str:
    if not raw_text:
        return ""
    return raw_text[-MAX_FALLBACK_TEXT_CHARS:]


def _format_benchmark_context(raw_text: Optional[str]) -> str:
    """
    Extract targeted text windows around benchmark/result keywords.

    Complex PDFs often produce unusable table rows while the surrounding text still
    contains benchmark names and values. This context is mainly for Sonnet fallback.
    """
    if not raw_text:
        return ""

    lowered = raw_text.lower()
    windows: list[tuple[int, int]] = []

    for keyword in BENCHMARK_CONTEXT_KEYWORDS:
        pattern = re.escape(keyword.lower())
        for match_index, match in enumerate(re.finditer(pattern, lowered)):
            if len(windows) >= MAX_BENCHMARK_CONTEXT_WINDOWS:
                break
            if match_index >= 4:
                break
            start = max(0, match.start() - BENCHMARK_CONTEXT_WINDOW_CHARS)
            end = min(len(raw_text), match.end() + BENCHMARK_CONTEXT_WINDOW_CHARS)
            windows.append((start, end))
        if len(windows) >= MAX_BENCHMARK_CONTEXT_WINDOWS:
            break

    if not windows:
        return _format_fallback_text(raw_text)

    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    parts = []
    total_chars = 0
    for index, (start, end) in enumerate(merged, start=1):
        chunk = raw_text[start:end].strip()
        if not chunk:
            continue
        section = f"[Benchmark context window {index}]\n{chunk}"
        if total_chars + len(section) > MAX_BENCHMARK_CONTEXT_CHARS:
            remaining = MAX_BENCHMARK_CONTEXT_CHARS - total_chars
            if remaining > 500:
                parts.append(section[:remaining] + "\n[truncated]")
            break
        parts.append(section)
        total_chars += len(section)

    return "\n\n---\n\n".join(parts)


def _has_complex_tables(tables: list) -> bool:
    return any(bool(table.get("needs_vision")) for table in tables)


def _should_include_fallback_text(
    *,
    tables: list,
    tables_truncated: bool,
    complex_tables: bool,
) -> bool:
    return not tables or tables_truncated or complex_tables


def _should_use_sonnet_fallback(
    *,
    benchmarks: list[BenchmarkResult],
    parse_error: Optional[str],
    tables: list,
    tables_truncated: bool,
    complex_tables: bool,
) -> bool:
    if benchmarks:
        return False
    return bool(parse_error or tables_truncated or complex_tables or not tables)


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
    *,
    model: str,
    context_label: str,
    proposed_method: str,
    tables_text: str,
    fallback_text: str,
) -> tuple[Optional[str], Optional[str]]:
    user_content = (
        f"Proposed method: {proposed_method}\n\n"
        f"## Extracted PDF tables with page context\n"
        f"{tables_text}"
    )

    if fallback_text:
        user_content += f"\n\n## Fallback paper text context\n{fallback_text}"

    try:
        response = _client.messages.create(
            model=model,
            max_tokens=2500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw, error = _extract_text_block(response, context=context_label)
        if error:
            return None, error
        logger.info("%s response: %d chars", context_label, len(raw or ""))
        return raw, None
    except Exception as exc:
        logger.exception("%s call failed", context_label)
        return None, f"{context_label} call failed: {exc}"


def _call_llm_repair(
    bad_json: str,
    *,
    model: str,
    context_label: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        response = _client.messages.create(
            model=model,
            max_tokens=2500,
            system="You are a JSON repair specialist. Return ONLY valid JSON array, no explanation.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The following JSON array is invalid. Fix it and return ONLY the corrected JSON array. "
                        "If the input is prose or does not contain a JSON array, return [].\n\n"
                        f"{bad_json[:4000]}"
                    ),
                }
            ],
        )
        return _extract_text_block(response, context=context_label)
    except Exception as exc:
        logger.exception("%s call failed", context_label)
        return None, f"{context_label} call failed: {exc}"


def _parse_benchmarks(raw_json: str) -> tuple[list[BenchmarkResult], Optional[str]]:
    cleaned = _strip_json_fences(raw_json)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return [], f"JSON parse error: {exc} | raw: {cleaned[:200]}"

    if not isinstance(data, list):
        return [], f"Expected JSON array, got {type(data).__name__}"

    results: list[BenchmarkResult] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        value = _parse_float(item.get("value"))
        if value is None:
            logger.warning("Skipping benchmark with non-numeric value: %r", item)
            continue

        task = str(item.get("task") or "").strip()
        metric = str(item.get("metric") or "").strip()

        if not task or not metric:
            logger.warning("Skipping benchmark with missing task/metric: %r", item)
            continue

        results.append(
            BenchmarkResult(
                task=task,
                metric=metric,
                value=value,
                unit=_as_optional_string(item.get("unit")),
                baseline_comparison=_as_optional_string(item.get("baseline_comparison")),
                conditions=_as_optional_string(item.get("conditions")),
            )
        )

    return results, None


def benchmark_analyst_agent(state: PaperIntelState) -> dict:
    """
    Benchmark Analyst Agent LangGraph node.
    Extracts benchmark results from PDF tables and page-level context.
    """
    logger.info("Benchmark agent started")

    raw_text = state.get("raw_text")
    pdf_path = state.get("pdf_path")
    text_by_page = state.get("text_by_page")
    proposed_method = _proposed_method_name(state)

    tables = []
    new_errors = []

    if pdf_path:
        try:
            tables = extract_tables(pdf_path)
            logger.info("Extracted %d tables from %s", len(tables), pdf_path)
        except Exception as exc:
            logger.warning("Could not extract tables from %s: %s", pdf_path, exc)
            new_errors.append(f"Benchmark: table extraction failed: {exc}")
    else:
        logger.warning("Benchmark: pdf_path missing, falling back to raw_text only")
        new_errors.append("Benchmark: pdf_path missing, using raw_text fallback")

    tables_text, tables_truncated = _format_tables_with_context(tables, text_by_page)
    complex_tables = _has_complex_tables(tables)
    fallback_text = (
        _format_benchmark_context(raw_text)
        if _should_include_fallback_text(
            tables=tables,
            tables_truncated=tables_truncated,
            complex_tables=complex_tables,
        )
        else ""
    )

    if not tables and not fallback_text:
        return paper_error(
            state,
            "; ".join(new_errors + ["Benchmark: no tables or text context available"]),
            "benchmark",
        )

    raw_json, llm_error = _call_llm(
        model=settings.haiku_model,
        context_label="Benchmark Haiku LLM",
        proposed_method=proposed_method,
        tables_text=tables_text,
        fallback_text=fallback_text,
    )

    if llm_error:
        return paper_error(
            state,
            "; ".join(new_errors + [llm_error]),
            "benchmark",
        )

    benchmarks, parse_error = _parse_benchmarks(raw_json or "")
    parse_warning = None

    if parse_error:
        logger.warning("Benchmark parse error: %s; attempting repair", parse_error)
        parse_warning = "Benchmark initial parse failed and required repair"
        repaired_json, repair_error = _call_llm_repair(
            raw_json or "",
            model=settings.haiku_model,
            context_label="Benchmark Haiku repair LLM",
        )

        if repair_error:
            logger.warning(
                "Benchmark Haiku repair failed; Sonnet fallback may run: %s",
                repair_error,
            )
            new_errors.append(
                f"Benchmark Haiku parse failed: {parse_error}; repair failed: {repair_error}"
            )
            parse_error = f"{parse_error}; repair failed: {repair_error}"
        else:
            benchmarks, parse_error = _parse_benchmarks(repaired_json or "")

    if _should_use_sonnet_fallback(
        benchmarks=benchmarks,
        parse_error=parse_error,
        tables=tables,
        tables_truncated=tables_truncated,
        complex_tables=complex_tables,
    ):
        logger.info(
            "Benchmark Sonnet fallback triggered: benchmarks=%d parse_error=%s "
            "tables=%d truncated=%s complex=%s",
            len(benchmarks),
            bool(parse_error),
            len(tables),
            tables_truncated,
            complex_tables,
        )
        fallback_warning = (
            "Benchmark Sonnet fallback used due to complex, truncated, empty, or "
            "unparseable benchmark context"
        )
        sonnet_fallback_text = fallback_text or _format_benchmark_context(raw_text)
        sonnet_json, sonnet_error = _call_llm(
            model=settings.sonnet_model,
            context_label="Benchmark Sonnet fallback LLM",
            proposed_method=proposed_method,
            tables_text=tables_text,
            fallback_text=sonnet_fallback_text,
        )

        if sonnet_error:
            new_errors.append(f"{fallback_warning}; failed: {sonnet_error}")
        else:
            sonnet_benchmarks, sonnet_parse_error = _parse_benchmarks(sonnet_json or "")

            if sonnet_parse_error:
                repaired_json, repair_error = _call_llm_repair(
                    sonnet_json or "",
                    model=settings.sonnet_model,
                    context_label="Benchmark Sonnet repair LLM",
                )
                if repair_error:
                    new_errors.append(
                        f"{fallback_warning}; parse failed: {sonnet_parse_error}; "
                        f"repair failed: {repair_error}"
                    )
                else:
                    sonnet_benchmarks, sonnet_parse_error = _parse_benchmarks(
                        repaired_json or ""
                    )

            if sonnet_parse_error:
                new_errors.append(
                    f"{fallback_warning}; parse failed after repair: {sonnet_parse_error}"
                )
            else:
                benchmarks = sonnet_benchmarks
                parse_error = None
                parse_warning = (
                    f"{parse_warning}; {fallback_warning}"
                    if parse_warning
                    else fallback_warning
                )

    if parse_error:
        return {
            "errors": new_errors + [f"Benchmark parse failed after repair: {parse_error}"],
            "benchmarks": [],
            "processing_stage": "readiness",
        }

    logger.info("Benchmark agent complete: %d results extracted", len(benchmarks))

    result = {
        "benchmarks": benchmarks,
        "processing_stage": "readiness",
    }
    final_errors = new_errors + ([parse_warning] if parse_warning else [])
    if final_errors:
        result["errors"] = final_errors
    return result
