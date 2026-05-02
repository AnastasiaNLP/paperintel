import json
import logging
import re
from pathlib import Path
from typing import Optional

from agents.error_utils import paper_error
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.errors import ErrorCodes, make_error
from models.schemas import (
    BenchmarkResult,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    ProductionReadiness,
)
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "report_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

VALID_DIFFICULTIES = {"trivial", "moderate", "significant", "research_only"}
VALID_ACTIONS = {"implement_now", "prototype", "watch", "skip"}

MAX_BENCHMARKS_IN_EVIDENCE = 15
MAX_ABSTRACT_CHARS = 1200
MAX_AUTHORS_IN_HEADER = 8


def _warning_errors(messages: list[str]) -> list:
    return [
        make_error(
            ErrorCodes.WARNING,
            message,
            node="report",
            severity="warning",
            recoverable=True,
        )
        for message in messages
    ]


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_text_block(
    response: object,
    context: str,
) -> tuple[Optional[str], Optional[str]]:
    content = getattr(response, "content", None)
    if not content:
        return None, f"{context}: empty content"

    block = content[0]
    raw = getattr(block, "text", None)
    if not isinstance(raw, str) or not raw.strip():
        return None, f"{context}: non-text block"

    return raw.strip(), None


def _build_evidence_json(
    metadata: Optional[PaperMetadata],
    extraction: Optional[MethodExtraction],
    benchmarks: list[BenchmarkResult],
    readiness: Optional[ProductionReadiness],
) -> str:
    evidence = {
        "paper": (
            {
                "title": metadata.title,
                "arxiv_id": metadata.arxiv_id,
                "published_date": metadata.published_date,
                "abstract": metadata.abstract[:MAX_ABSTRACT_CHARS],
                "citation_count": metadata.citation_count,
            }
            if metadata
            else None
        ),
        "method": (
            {
                "method_name": extraction.method_name,
                "description": extraction.description,
                "novelty_claim": extraction.novelty_claim,
                "key_components": extraction.key_components,
                "compared_to": extraction.compared_to,
                "limitations_stated": extraction.limitations_stated,
            }
            if extraction
            else None
        ),
        "benchmarks": [
            {
                "task": benchmark.task,
                "metric": benchmark.metric,
                "value": benchmark.value,
                "unit": benchmark.unit,
                "baseline_comparison": benchmark.baseline_comparison,
                "conditions": benchmark.conditions,
            }
            for benchmark in benchmarks[:MAX_BENCHMARKS_IN_EVIDENCE]
        ],
        "production_readiness": (
            {
                "has_open_code": readiness.has_open_code,
                "code_url": readiness.code_url,
                "huggingface_model": readiness.huggingface_model,
                "framework_integrations": readiness.framework_integrations,
                "min_gpu_requirement": readiness.min_gpu_requirement,
                "estimated_inference_cost": readiness.estimated_inference_cost,
                "dependencies": readiness.dependencies,
                "maturity_level": readiness.maturity_level,
                "maturity_reasoning": readiness.maturity_reasoning,
            }
            if readiness
            else None
        ),
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _call_llm(evidence_json: str) -> tuple[Optional[str], Optional[str]]:
    return call_text_llm(
        requested_model=settings.haiku_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=evidence_json,
        max_tokens=1200,
        context_label="Report LLM",
    )


def _call_llm_repair(bad_json: str) -> tuple[Optional[str], Optional[str]]:
    return call_text_llm(
        requested_model=settings.haiku_model,
        system_prompt=(
            "You are a JSON repair specialist. Return ONLY a valid JSON object. "
            'The first character must be "{". The last character must be "}". '
            "No prose, markdown, or explanation."
        ),
        user_content=(
            "Fix invalid JSON and return only the JSON object:\n\n"
            f"{bad_json[:3000]}"
        ),
        max_tokens=1200,
        context_label="Report repair",
    )


def _parse_claims(raw_json: str) -> tuple[Optional[dict], Optional[str]]:
    cleaned = _strip_fences(raw_json)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected object, got {type(data).__name__}"

    return data, None


def _fallback_executive_summary(
    metadata: Optional[PaperMetadata],
    extraction: Optional[MethodExtraction],
    readiness: Optional[ProductionReadiness],
    action: str,
) -> str:
    title = metadata.title if metadata else "This paper"
    method = extraction.method_name if extraction else "the proposed method"
    maturity = readiness.maturity_level if readiness else "unknown"
    return (
        f"{title} proposes {method}. Maturity: {maturity}. "
        f"Automated summary unavailable; recommended action is '{action}' "
        f"based on available evidence. See sections below for extracted details."
    )


def _normalize_engineer_report(
    claims: dict,
    metadata: Optional[PaperMetadata],
    extraction: Optional[MethodExtraction],
    benchmarks: list[BenchmarkResult],
    readiness: Optional[ProductionReadiness],
) -> tuple[Optional[EngineerReport], Optional[str]]:
    difficulty = str(claims.get("implementation_difficulty") or "").strip()
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "research_only"

    action = str(claims.get("recommended_action") or "").strip()
    if action not in VALID_ACTIONS:
        action = "skip"

    executive_summary = str(claims.get("executive_summary") or "").strip()
    key_innovation = str(claims.get("key_innovation") or "").strip()
    practical_implications = str(claims.get("practical_implications") or "").strip()
    action_reasoning = str(claims.get("action_reasoning") or "").strip()

    if extraction is None:
        action = "skip"
        difficulty = "research_only"
        if "insufficient extraction" not in action_reasoning.lower():
            action_reasoning = (
                "Insufficient extraction: method details unavailable, cannot assess. "
                + action_reasoning
            ).strip()

    if readiness is None:
        difficulty = "research_only"
        if action in {"implement_now", "prototype"}:
            original = action
            action = "watch"
            action_reasoning = (
                f"Downgraded from {original}: production_readiness unavailable. "
                + action_reasoning
            ).strip()

        lower_reasoning = action_reasoning.lower()
        if (
            "production_readiness unavailable" not in lower_reasoning
            and "production readiness unavailable" not in lower_reasoning
        ):
            action_reasoning = (
                "Production readiness unavailable. " + action_reasoning
            ).strip()

    if readiness is not None and readiness.maturity_level == "research_only":
        difficulty = "research_only"
        if action in {"implement_now", "prototype"}:
            original = action
            action = "watch"
            action_reasoning = (
                f"Downgraded from {original}: maturity is research_only. "
                + action_reasoning
            ).strip()
        if "research_only" not in action_reasoning.lower():
            action_reasoning = ("Maturity is research_only. " + action_reasoning).strip()

    if readiness is not None and readiness.maturity_level == "experimental":
        if action == "implement_now":
            action = "prototype"
            action_reasoning = (
                "Downgraded from implement_now: maturity is experimental. "
                + action_reasoning
            ).strip()
        if difficulty == "trivial":
            difficulty = "moderate"

    if not benchmarks:
        if action == "implement_now":
            action = "prototype"
            action_reasoning = (
                "Downgraded from implement_now: no benchmarks extracted. "
                + action_reasoning
            ).strip()
        if "no benchmarks extracted" not in action_reasoning.lower():
            action_reasoning = ("No benchmarks extracted. " + action_reasoning).strip()

    if not executive_summary:
        executive_summary = _fallback_executive_summary(
            metadata,
            extraction,
            readiness,
            action,
        )

    try:
        return (
            EngineerReport(
                executive_summary=executive_summary,
                key_innovation=key_innovation or "Not extracted.",
                practical_implications=practical_implications or "Not extracted.",
                implementation_difficulty=difficulty,
                recommended_action=action,
                action_reasoning=action_reasoning or "Not provided.",
            ),
            None,
        )
    except Exception as exc:
        return None, f"EngineerReport validation error: {exc}"


def _na(value: Optional[str]) -> str:
    return value if value and str(value).strip() else "_not available_"


def _md_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_benchmark_table(benchmarks: list[BenchmarkResult]) -> str:
    if not benchmarks:
        return "_No benchmarks extracted from the paper._"

    lines = [
        "| Task | Metric | Value | Unit | Baseline | Conditions |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for benchmark in benchmarks:
        lines.append(
            "| {task} | {metric} | {value} | {unit} | {baseline} | {cond} |".format(
                task=_md_cell(benchmark.task),
                metric=_md_cell(benchmark.metric),
                value=_md_cell(benchmark.value),
                unit=_md_cell(benchmark.unit),
                baseline=_md_cell(benchmark.baseline_comparison),
                cond=_md_cell(benchmark.conditions),
            )
        )

    return "\n".join(lines)


def _render_readiness_section(readiness: Optional[ProductionReadiness]) -> str:
    if readiness is None:
        return "_Production readiness assessment unavailable._"

    frameworks = ", ".join(readiness.framework_integrations) or "_none detected_"
    deps = ", ".join(readiness.dependencies[:15]) or "_not detected_"
    if len(readiness.dependencies) > 15:
        deps += f", ... (+{len(readiness.dependencies) - 15} more)"

    hf_line = (
        f"[{readiness.huggingface_model}](https://huggingface.co/{readiness.huggingface_model})"
        if readiness.huggingface_model
        else "_not available_"
    )
    code_line = (
        f"yes - [{readiness.code_url}]({readiness.code_url})"
        if readiness.has_open_code and readiness.code_url
        else ("yes" if readiness.has_open_code else "no")
    )

    return (
        f"- **Maturity:** `{readiness.maturity_level}`\n"
        f"- **Open code:** {code_line}\n"
        f"- **HuggingFace model:** {hf_line}\n"
        f"- **Framework integrations:** {frameworks}\n"
        f"- **Min GPU (inference):** {_na(readiness.min_gpu_requirement)}\n"
        f"- **Estimated inference cost:** {_na(readiness.estimated_inference_cost)}\n"
        f"- **Dependencies:** {deps}\n"
        f"- **Reasoning:** {readiness.maturity_reasoning or '_not provided_'}"
    )


def _render_method_section(extraction: Optional[MethodExtraction]) -> str:
    if extraction is None:
        return "_Method extraction unavailable._"

    key_components = (
        "\n".join(f"- {component}" for component in extraction.key_components)
        if extraction.key_components
        else "- _not extracted_"
    )
    compared_to = ", ".join(extraction.compared_to) or "_not stated_"
    limitations = (
        "\n".join(f"- {limitation}" for limitation in extraction.limitations_stated)
        if extraction.limitations_stated
        else "- _none stated by authors_"
    )

    return (
        f"**Method:** {extraction.method_name}\n\n"
        f"{extraction.description}\n\n"
        f"**Novelty claim:** {extraction.novelty_claim}\n\n"
        f"**Key components:**\n{key_components}\n\n"
        f"**Baselines compared to:** {compared_to}\n\n"
        f"**Limitations stated by authors:**\n{limitations}"
    )


def _render_authors(metadata: Optional[PaperMetadata]) -> str:
    if not metadata or not metadata.authors:
        return "_unknown_"

    authors = metadata.authors
    if len(authors) > MAX_AUTHORS_IN_HEADER:
        return (
            ", ".join(authors[:MAX_AUTHORS_IN_HEADER])
            + f", ... (+{len(authors) - MAX_AUTHORS_IN_HEADER} more)"
        )

    return ", ".join(authors)


def _render_markdown_report(
    metadata: Optional[PaperMetadata],
    extraction: Optional[MethodExtraction],
    benchmarks: list[BenchmarkResult],
    readiness: Optional[ProductionReadiness],
    engineer_report: EngineerReport,
) -> str:
    title = metadata.title if metadata else "Untitled paper"
    authors = _render_authors(metadata)
    arxiv_id = metadata.arxiv_id if metadata else None
    published = metadata.published_date if metadata else "_unknown_"
    arxiv_line = (
        f"- **arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n"
        if arxiv_id
        else ""
    )

    return (
        f"# {title}\n\n"
        f"- **Authors:** {authors}\n"
        f"- **Published:** {published}\n"
        f"{arxiv_line}"
        f"\n"
        f"## Verdict\n\n"
        f"**Recommendation:** `{engineer_report.recommended_action}`  \n"
        f"**Implementation difficulty:** `{engineer_report.implementation_difficulty}`\n\n"
        f"{engineer_report.action_reasoning}\n\n"
        f"## Executive summary\n\n"
        f"{engineer_report.executive_summary}\n\n"
        f"## Key innovation\n\n"
        f"{engineer_report.key_innovation}\n\n"
        f"## Practical implications\n\n"
        f"{engineer_report.practical_implications}\n\n"
        f"## Method\n\n"
        f"{_render_method_section(extraction)}\n\n"
        f"## Benchmarks\n\n"
        f"{_render_benchmark_table(benchmarks)}\n\n"
        f"## Production readiness\n\n"
        f"{_render_readiness_section(readiness)}\n"
    )


def report_agent(state: PaperIntelState) -> dict:
    """
    Report Generator Agent.

    Synthesizes upstream outputs into:
    - a structured EngineerReport
    - a deterministic Markdown rendering

    Graceful degradation:
    - metadata, extraction, benchmarks, and production_readiness may each be missing.
    - missing sections render as not available.
    - verdict is forced to skip if extraction is missing.
    - verdict is capped at watch if readiness is missing or maturity is research_only.
    - empty executive_summary falls back to a deterministic summary.
    """
    logger.info("Report agent started")

    metadata = state.get("metadata")
    extraction = state.get("method_extraction")
    benchmarks = state.get("benchmarks", []) or []
    readiness = state.get("production_readiness")

    degradation_notes: list[str] = []
    if extraction is None:
        degradation_notes.append("Report warning: method_extraction missing")
    if not benchmarks:
        degradation_notes.append("Report warning: benchmarks missing or empty")
    if readiness is None:
        degradation_notes.append("Report warning: production_readiness missing")
    if degradation_notes:
        logger.warning("Report degradation: %s", "; ".join(degradation_notes))

    evidence_json = _build_evidence_json(metadata, extraction, benchmarks, readiness)

    raw, llm_error = _call_llm(evidence_json)
    if llm_error:
        return paper_error(state, llm_error, "report")

    claims, parse_error = _parse_claims(raw or "")
    if parse_error:
        repaired, repair_error = _call_llm_repair(raw or "")
        if repair_error:
            return paper_error(
                state,
                f"Report parse failed: {parse_error}; repair: {repair_error}",
                "report",
            )
        claims, parse_error = _parse_claims(repaired or "")

    if parse_error or claims is None:
        return paper_error(
            state,
            f"Report parse failed after repair: {parse_error}",
            "report",
        )

    engineer_report, norm_error = _normalize_engineer_report(
        claims,
        metadata,
        extraction,
        benchmarks,
        readiness,
    )
    if norm_error or engineer_report is None:
        return paper_error(
            state,
            norm_error or "Report normalization failed",
            "report",
        )

    markdown = _render_markdown_report(
        metadata,
        extraction,
        benchmarks,
        readiness,
        engineer_report,
    )

    logger.info(
        "Report complete: action=%s difficulty=%s md_chars=%d",
        engineer_report.recommended_action,
        engineer_report.implementation_difficulty,
        len(markdown),
    )

    result: dict = {
        "engineer_report": engineer_report,
        "full_markdown_report": markdown,
        "processing_stage": "completed",
    }
    if degradation_notes:
        result["errors"] = _warning_errors(degradation_notes)

    return result
