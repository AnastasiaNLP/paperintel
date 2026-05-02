import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agents.llm_provider import call_text_llm
from config.settings import settings
from models.errors import ErrorCodes, make_error
from models.schemas import (
    BenchmarkResult,
    ComparisonMatrixRow,
    ComparisonReport,
    ConstraintRecommendation,
    PaperSlot,
)
from models.state import PaperIntelState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "comparator_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
MAX_RECOMMENDATIONS = 5
VALID_WINNER_BASES = {
    "readiness_dominant",
    "benchmark_dominant",
    "mixed",
    "no_clear_winner",
}


def _warning_error(message: str) -> object:
    return make_error(
        ErrorCodes.WARNING,
        message,
        node="comparator",
        severity="warning",
        recoverable=True,
    )

LOWER_IS_BETTER_EXACT_METRICS = {
    "latency",
    "latency_ms",
    "runtime",
    "time",
    "cost",
    "cost_per_token",
    "perplexity",
    "ppl",
    "error_rate",
    "wer",
    "cer",
    "loss",
}

LOWER_IS_BETTER_HINTS = {
    "latency",
    "runtime",
    "perplexity",
    "error_rate",
    "wer",
    "cer",
    "loss",
}

TASK_ALIASES = {
    "math500": "math-500",
    "math 500": "math-500",
    "math-500": "math-500",
    "humaneval": "humaneval",
    "human eval": "humaneval",
    "mmlu": "mmlu",
    "mmlu pro": "mmlu-pro",
    "mmlu-pro": "mmlu-pro",
    "gsm8k": "gsm8k",
    "aime": "aime",
    "aime 2024": "aime-2024",
    "gpqa": "gpqa",
    "gpqa diamond": "gpqa-diamond",
}

METRIC_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "pass@1": "pass@1",
    "pass @ 1": "pass@1",
    "f1": "f1",
    "bleu": "bleu",
    "rouge": "rouge",
    "exact match": "exact_match",
    "em": "exact_match",
}

CONDITION_ALIASES = {
    "0 shot": "zero-shot",
    "0-shot": "zero-shot",
    "zero shot": "zero-shot",
    "zero-shot": "zero-shot",
    "1 shot": "1-shot",
    "1-shot": "1-shot",
    "5 shot": "5-shot",
    "5-shot": "5-shot",
    "five shot": "5-shot",
    "greedy decoding": "greedy",
    "greedy": "greedy",
}

UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
}


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _trim_to_json_object(text: str) -> str:
    cleaned = _strip_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _as_paper_slots(value: object) -> list[PaperSlot]:
    if not isinstance(value, list):
        return []

    slots: list[PaperSlot] = []
    for item in value:
        if isinstance(item, PaperSlot):
            slots.append(item)
        elif isinstance(item, dict):
            try:
                slots.append(PaperSlot(**item))
            except Exception as exc:
                logger.warning("Skipping invalid PaperSlot dict: %s", exc)
    return slots


def _normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_/]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_task_name(task: str) -> str:
    normalized = _normalize_name(task)
    compact = normalized.replace(" ", "")
    if normalized in TASK_ALIASES:
        return TASK_ALIASES[normalized]
    if compact in TASK_ALIASES:
        return TASK_ALIASES[compact]
    return normalized.replace(" ", "-")


def _normalize_metric_name(metric: str) -> str:
    normalized = _normalize_name(metric)
    return METRIC_ALIASES.get(normalized, normalized.replace(" ", "_"))


def _normalize_condition_name(condition: Optional[str]) -> str:
    if not condition:
        return ""
    normalized = _normalize_name(condition)
    return CONDITION_ALIASES.get(normalized, normalized.replace(" ", "-"))


def _normalize_unit_name(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    normalized = _normalize_name(unit)
    return UNIT_ALIASES.get(normalized, normalized)


def _is_higher_better(metric: str) -> bool:
    normalized = _normalize_metric_name(metric)
    if normalized in LOWER_IS_BETTER_EXACT_METRICS:
        return False
    return not any(hint in normalized for hint in LOWER_IS_BETTER_HINTS)


def _comparison_key(benchmark: BenchmarkResult) -> tuple[str, str]:
    return _normalize_task_name(benchmark.task), _normalize_metric_name(benchmark.metric)


def _select_representative_benchmark(
    benchmarks: list[BenchmarkResult],
    *,
    higher_is_better: bool,
) -> tuple[Optional[BenchmarkResult], bool, str]:
    """
    Choose a representative value for display/comparison.

    This may downgrade row comparability when multiple variants exist. The
    representative is not an authoritative canonical result for the paper.
    """
    if not benchmarks:
        return None, True, ""

    if len(benchmarks) == 1:
        return benchmarks[0], True, ""

    condition_keys = {_normalize_condition_name(item.conditions) for item in benchmarks}
    if len(condition_keys) > 1:
        return (
            benchmarks[0],
            False,
            "Multiple results for same task/metric with different conditions; "
            "representative_selection=first_seen.",
        )

    representative = sorted(
        benchmarks,
        key=lambda item: item.value,
        reverse=higher_is_better,
    )[0]

    return (
        representative,
        False,
        "Multiple results for same task/metric/condition; "
        "representative_selection=best_value_by_metric_direction.",
    )


def _winner_for_values(
    values_by_paper: dict[int, Optional[float]],
    *,
    higher_is_better: bool,
    is_comparable: bool,
) -> tuple[Optional[int], Optional[float]]:
    if not is_comparable:
        return None, None

    numeric_values = {
        paper_index: value
        for paper_index, value in values_by_paper.items()
        if value is not None
    }
    if len(numeric_values) < 2:
        return None, None

    sorted_values = sorted(
        numeric_values.items(),
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    winner_index, winner_value = sorted_values[0]
    second_value = sorted_values[1][1]

    if winner_value == second_value:
        return None, 0.0

    margin = winner_value - second_value if higher_is_better else second_value - winner_value
    return winner_index, margin


def _build_comparison_matrix(papers: list[PaperSlot]) -> list[ComparisonMatrixRow]:
    paper_indexes = [paper.paper_index for paper in papers]
    grouped: dict[tuple[str, str], dict[int, list[BenchmarkResult]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for paper in papers:
        for benchmark in paper.benchmarks:
            grouped[_comparison_key(benchmark)][paper.paper_index].append(benchmark)

    rows: list[ComparisonMatrixRow] = []

    for (task, metric), by_paper in sorted(grouped.items()):
        higher_is_better = _is_higher_better(metric)
        values_by_paper: dict[int, Optional[float]] = {}
        units_by_paper: dict[int, Optional[str]] = {}
        conditions_by_paper: dict[int, Optional[str]] = {}
        duplicate_counts_by_paper: dict[int, int] = {}
        notes: list[str] = []
        is_comparable = True

        for paper_index in paper_indexes:
            variants = by_paper.get(paper_index, [])
            # 0 means this paper did not report this task/metric row.
            duplicate_counts_by_paper[paper_index] = len(variants)

            representative, comparable_within_paper, note = _select_representative_benchmark(
                variants,
                higher_is_better=higher_is_better,
            )

            if note:
                notes.append(f"Paper {paper_index}: {note}")
            if not comparable_within_paper:
                is_comparable = False

            if representative is None:
                values_by_paper[paper_index] = None
                units_by_paper[paper_index] = None
                conditions_by_paper[paper_index] = None
            else:
                values_by_paper[paper_index] = representative.value
                units_by_paper[paper_index] = representative.unit
                conditions_by_paper[paper_index] = representative.conditions

        present_conditions = {
            _normalize_condition_name(value)
            for value in conditions_by_paper.values()
            if value
        }
        if len(present_conditions) > 1:
            is_comparable = False
            notes.append("Compared papers report this task/metric under different conditions.")

        present_units = {
            _normalize_unit_name(value)
            for value in units_by_paper.values()
            if value
        }
        if len(present_units) > 1:
            is_comparable = False
            notes.append(
                "Compared papers report this task/metric with different units. "
                "Future normalization may handle percent/fraction equivalence."
            )

        winner_index, winner_margin = _winner_for_values(
            values_by_paper,
            higher_is_better=higher_is_better,
            is_comparable=is_comparable,
        )

        rows.append(
            ComparisonMatrixRow(
                task=task,
                metric=metric,
                values_by_paper=values_by_paper,
                units_by_paper=units_by_paper,
                conditions_by_paper=conditions_by_paper,
                duplicate_counts_by_paper=duplicate_counts_by_paper,
                winner_index=winner_index,
                winner_margin=winner_margin,
                higher_is_better=higher_is_better,
                is_comparable=is_comparable,
                comparability_notes=" ".join(notes) or None,
            )
        )

    return rows


def _build_unique_tasks_per_paper(
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
) -> dict[int, list[str]]:
    unique = {paper.paper_index: [] for paper in papers}
    task_to_papers: dict[str, set[int]] = defaultdict(set)

    for row in matrix:
        present = {
            paper_index
            for paper_index, value in row.values_by_paper.items()
            if value is not None
        }
        task_to_papers[row.task].update(present)

    for task, present_papers in task_to_papers.items():
        if len(present_papers) == 1:
            paper_index = next(iter(present_papers))
            unique[paper_index].append(task)

    return {paper_index: sorted(set(tasks)) for paper_index, tasks in unique.items()}


def _build_unique_rows_per_paper(
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
) -> dict[int, list[str]]:
    unique = {paper.paper_index: [] for paper in papers}

    for row in matrix:
        present = [
            paper_index
            for paper_index, value in row.values_by_paper.items()
            if value is not None
        ]
        if len(present) == 1:
            unique[present[0]].append(f"{row.task}/{row.metric}")

    return {paper_index: sorted(set(rows)) for paper_index, rows in unique.items()}


def _build_matrix_stats(
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
) -> dict:
    paper_indexes = {paper.paper_index for paper in papers}
    comparable_rows = sum(1 for row in matrix if row.is_comparable)
    rows_with_winner = sum(1 for row in matrix if row.winner_index is not None)
    rows_with_overlap = sum(
        1
        for row in matrix
        if sum(1 for value in row.values_by_paper.values() if value is not None) >= 2
    )
    wins_by_paper = {paper_index: 0 for paper_index in paper_indexes}
    for row in matrix:
        if row.winner_index is not None:
            wins_by_paper[row.winner_index] = wins_by_paper.get(row.winner_index, 0) + 1

    overlap_ratio = rows_with_overlap / len(matrix) if matrix else 0.0

    return {
        "rows": len(matrix),
        "comparable_rows": comparable_rows,
        "rows_with_winner": rows_with_winner,
        "benchmark_overlap_ratio": round(overlap_ratio, 4),
        "wins_by_paper": wins_by_paper,
    }


def _build_papers_summary(papers: list[PaperSlot]) -> list[dict]:
    summaries = []

    for paper in papers:
        metadata = paper.metadata
        extraction = paper.method_extraction
        readiness = paper.production_readiness
        report = paper.engineer_report

        summaries.append(
            {
                "paper_index": paper.paper_index,
                "input_url": paper.input_url,
                "completed": paper.completed,
                "title": metadata.title if metadata else None,
                "arxiv_id": metadata.arxiv_id if metadata else None,
                "citation_count": metadata.citation_count if metadata else None,
                "method_name": extraction.method_name if extraction else None,
                "limitations_stated": extraction.limitations_stated if extraction else [],
                "maturity_level": readiness.maturity_level if readiness else None,
                "has_open_code": readiness.has_open_code if readiness else False,
                "code_url": readiness.code_url if readiness else None,
                "huggingface_model": readiness.huggingface_model if readiness else None,
                "framework_integrations": readiness.framework_integrations if readiness else [],
                "min_gpu_requirement": readiness.min_gpu_requirement if readiness else None,
                "dependencies": readiness.dependencies[:15] if readiness else [],
                "recommended_action": report.recommended_action if report else None,
                "implementation_difficulty": report.implementation_difficulty if report else None,
                "benchmark_count": len(paper.benchmarks),
                "errors": paper.errors,
            }
        )

    return summaries


def _eligible_indexes(papers: list[PaperSlot]) -> set[int]:
    completed = {paper.paper_index for paper in papers if paper.completed}
    if completed:
        return completed
    return {paper.paper_index for paper in papers}


def _build_evidence_json(
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
    unique_tasks: dict[int, list[str]],
    unique_rows: dict[int, list[str]],
    papers_summary: list[dict],
    matrix_stats: dict,
) -> str:
    evidence = {
        "eligible_paper_indexes": sorted(_eligible_indexes(papers)),
        "papers_summary": papers_summary,
        "comparison_matrix": [row.model_dump() for row in matrix],
        "unique_tasks_per_paper": unique_tasks,
        "unique_rows_per_paper": unique_rows,
        "matrix_stats": matrix_stats,
        "readiness_reasoning": {
            paper.paper_index: (
                paper.production_readiness.maturity_reasoning
                if paper.production_readiness
                else None
            )
            for paper in papers
        },
        "single_paper_report_reasoning": {
            paper.paper_index: (
                paper.engineer_report.action_reasoning
                if paper.engineer_report
                else None
            )
            for paper in papers
        },
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _extract_text_block(response: object, context: str) -> tuple[Optional[str], Optional[str]]:
    content = getattr(response, "content", None)
    if not content:
        return None, f"{context}: empty content"
    block = content[0]
    raw = getattr(block, "text", None)
    if not isinstance(raw, str) or not raw.strip():
        return None, f"{context}: non-text block"
    return raw.strip(), None


def _comparator_model() -> str:
    return getattr(settings, "comparator_model", settings.haiku_model)


def _call_llm(evidence_json: str) -> tuple[Optional[str], Optional[str]]:
    return call_text_llm(
        requested_model=_comparator_model(),
        system_prompt=_SYSTEM_PROMPT,
        user_content=evidence_json,
        max_tokens=1400,
        context_label="Comparator LLM",
    )


def _call_llm_repair(bad_json: str) -> tuple[Optional[str], Optional[str]]:
    return call_text_llm(
        requested_model=_comparator_model(),
        system_prompt=(
            "You are a JSON repair specialist. Return ONLY a valid JSON object. "
            'The first character must be "{". The last character must be "}". '
            "No prose, markdown, or explanation."
        ),
        user_content=(
            "Fix invalid JSON and return only the JSON object:\n\n"
            f"{bad_json[:3000]}"
        ),
        max_tokens=1000,
        context_label="Comparator repair",
    )


def _parse_claims(raw_json: str) -> tuple[Optional[dict], Optional[str]]:
    cleaned = _trim_to_json_object(raw_json)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
    if not isinstance(data, dict):
        return None, f"Expected JSON object, got {type(data).__name__}"
    return data, None


def _normalize_constraint(value: str) -> str:
    normalized = _normalize_name(value)
    aliases = {
        "highest accuracy": "best accuracy",
        "top accuracy": "best accuracy",
        "best reasoning prototype": "best prototype for reasoning quality",
        "strongest reasoning prototype": "best prototype for reasoning quality",
        "best prototype for reasoning": "best prototype for reasoning quality",
        "best benchmark": "best benchmark evidence",
        "best benchmarks": "best benchmark evidence",
        "easiest deployment": "easiest to deploy",
        "deployment ease": "easiest to deploy",
    }
    return aliases.get(normalized, normalized)


def _normalize_recommendations(
    raw_recommendations: object,
    valid_indexes: set[int],
) -> list[ConstraintRecommendation]:
    if not isinstance(raw_recommendations, list):
        return []

    recommendations = []
    seen_constraints = set()

    for item in raw_recommendations:
        if not isinstance(item, dict):
            continue

        try:
            paper_index = int(item.get("recommended_paper_index"))
        except (TypeError, ValueError):
            continue

        if paper_index not in valid_indexes:
            logger.warning("Dropping comparator recommendation with invalid paper_index=%s", paper_index)
            continue

        constraint = _normalize_constraint(str(item.get("constraint") or "").strip())
        reasoning = str(item.get("reasoning") or "").strip()

        if not constraint or not reasoning or constraint in seen_constraints:
            continue

        recommendations.append(
            ConstraintRecommendation(
                constraint=constraint,
                recommended_paper_index=paper_index,
                reasoning=reasoning,
            )
        )
        seen_constraints.add(constraint)

        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    return recommendations


def _fallback_tradeoffs(papers: list[PaperSlot], matrix: list[ComparisonMatrixRow]) -> str:
    completed = [paper for paper in papers if paper.completed]
    if not completed:
        return "No completed papers were available for comparison."

    comparable_rows = sum(1 for row in matrix if row.is_comparable)
    non_comparable_rows = sum(1 for row in matrix if not row.is_comparable)

    if not matrix:
        return (
            f"Compared {len(completed)} completed papers, but no aligned benchmark rows were available. "
            "The comparison is qualitative and should rely on maturity, code/model availability, and implementation difficulty."
        )

    return (
        f"Compared {len(completed)} completed papers across {len(matrix)} benchmark rows "
        f"({comparable_rows} comparable, {non_comparable_rows} non-comparable). "
        "Missing or non-comparable benchmark evidence reduces confidence in any overall winner."
    )


def _fallback_overall_reasoning(
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
) -> str:
    completed = [paper for paper in papers if paper.completed]
    if not completed:
        return "No completed papers were available, so no reliable overall winner can be selected."

    maturity_counts = {}
    for paper in completed:
        maturity = (
            paper.production_readiness.maturity_level
            if paper.production_readiness
            else "unknown"
        )
        maturity_counts[maturity] = maturity_counts.get(maturity, 0) + 1

    return (
        f"The comparison includes {len(completed)} completed papers with maturity distribution "
        f"{maturity_counts} and {len(matrix)} benchmark rows. No clear overall winner was selected "
        "because the available evidence requires constraint-specific interpretation."
    )


def _paper_maturity_rank(paper: PaperSlot) -> int:
    maturity = (
        paper.production_readiness.maturity_level
        if paper.production_readiness
        else "unknown"
    )
    return {
        "production_ready": 3,
        "experimental": 2,
        "research_only": 1,
    }.get(maturity, 0)


def _determine_winner_basis(
    *,
    claims: dict,
    papers: list[PaperSlot],
    overall_winner_index: Optional[int],
    matrix_stats: dict,
) -> str:
    if overall_winner_index is None:
        return "no_clear_winner"

    claimed = str(claims.get("winner_basis") or "").strip()
    if claimed in VALID_WINNER_BASES and claimed != "no_clear_winner":
        return claimed

    rows_with_winner = int(matrix_stats.get("rows_with_winner") or 0)
    wins_by_paper = matrix_stats.get("wins_by_paper") or {}
    winner_row_wins = int(wins_by_paper.get(overall_winner_index, 0))

    if rows_with_winner == 0 or winner_row_wins == 0:
        return "readiness_dominant"

    max_row_wins = max((int(value) for value in wins_by_paper.values()), default=0)
    winner_has_top_benchmark_wins = winner_row_wins == max_row_wins

    winner = next((paper for paper in papers if paper.paper_index == overall_winner_index), None)
    if winner is None:
        return "no_clear_winner"

    winner_maturity = _paper_maturity_rank(winner)
    max_maturity = max((_paper_maturity_rank(paper) for paper in papers), default=0)
    winner_has_top_maturity = winner_maturity == max_maturity

    if winner_has_top_benchmark_wins and not winner_has_top_maturity:
        return "benchmark_dominant"
    if winner_has_top_benchmark_wins and winner_has_top_maturity:
        return "mixed"
    return "readiness_dominant"


def _normalize_comparison_report(
    claims: dict,
    papers: list[PaperSlot],
    matrix: list[ComparisonMatrixRow],
    unique_tasks: dict[int, list[str]],
    unique_rows: dict[int, list[str]],
    papers_summary: list[dict],
    matrix_stats: dict,
) -> ComparisonReport:
    valid_indexes = _eligible_indexes(papers)

    trade_offs = str(claims.get("trade_offs") or "").strip()
    if not trade_offs:
        trade_offs = _fallback_tradeoffs(papers, matrix)

    recommendations = _normalize_recommendations(
        claims.get("recommendations"),
        valid_indexes,
    )

    winner_raw = claims.get("overall_winner_index")
    overall_winner_index = None
    if winner_raw is not None:
        try:
            winner = int(winner_raw)
            if winner in valid_indexes:
                overall_winner_index = winner
            else:
                logger.warning("Dropping invalid overall_winner_index=%s", winner)
        except (TypeError, ValueError):
            logger.warning("Dropping non-integer overall_winner_index=%r", winner_raw)

    overall_reasoning = str(claims.get("overall_winner_reasoning") or "").strip()
    if not overall_reasoning:
        overall_reasoning = _fallback_overall_reasoning(papers, matrix)

    winner_basis = _determine_winner_basis(
        claims=claims,
        papers=papers,
        overall_winner_index=overall_winner_index,
        matrix_stats=matrix_stats,
    )

    return ComparisonReport(
        papers_summary=papers_summary,
        comparison_matrix=matrix,
        unique_tasks_per_paper=unique_tasks,
        unique_rows_per_paper=unique_rows,
        comparable_rows=int(matrix_stats.get("comparable_rows") or 0),
        rows_with_winner=int(matrix_stats.get("rows_with_winner") or 0),
        benchmark_overlap_ratio=float(matrix_stats.get("benchmark_overlap_ratio") or 0.0),
        wins_by_paper=matrix_stats.get("wins_by_paper") or {},
        winner_basis=winner_basis,
        trade_offs=trade_offs,
        recommendations=recommendations,
        overall_winner_index=overall_winner_index,
        overall_winner_reasoning=overall_reasoning,
    )


def _md_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_value(value: Optional[float], unit: Optional[str]) -> str:
    if value is None:
        return ""
    return f"{value} {unit}" if unit else str(value)


def _render_comparison_markdown(report: ComparisonReport) -> str:
    lines = ["# Paper Comparison", ""]

    lines.append("## Papers")
    for item in report.papers_summary:
        title = item.get("title") or f"Paper {item.get('paper_index')}"
        completed = "completed" if item.get("completed") else "incomplete"
        lines.append(
            f"- **{item.get('paper_index')}**: {_md_cell(title)} "
            f"(`{item.get('maturity_level')}`, `{item.get('recommended_action')}`, {completed})"
        )

    lines.extend(["", "## Benchmark Matrix", ""])
    lines.append(
        f"_Comparable rows: {report.comparable_rows}; rows with winner: "
        f"{report.rows_with_winner}; overlap ratio: {report.benchmark_overlap_ratio:.2f}._"
    )
    lines.append("")
    if not report.comparison_matrix:
        lines.append("_No aligned benchmarks available._")
    else:
        paper_indexes = [item["paper_index"] for item in report.papers_summary]
        header = "| Task | Metric | " + " | ".join(f"Paper {i}" for i in paper_indexes) + " | Winner | Notes |"
        sep = "| --- | --- | " + " | ".join("---:" for _ in paper_indexes) + " | --- | --- |"
        lines.extend([header, sep])

        for row in report.comparison_matrix:
            values = []
            for paper_index in paper_indexes:
                value = row.values_by_paper.get(paper_index)
                unit = row.units_by_paper.get(paper_index)
                values.append(_md_cell(_format_value(value, unit)))

            if row.winner_index is None:
                winner = ""
            elif row.winner_margin is None:
                winner = f"Paper {row.winner_index}"
            else:
                winner = f"Paper {row.winner_index} (+{row.winner_margin:g})"

            notes = row.comparability_notes or ("" if row.is_comparable else "Not directly comparable")
            lines.append(
                f"| {_md_cell(row.task)} | {_md_cell(row.metric)} | "
                + " | ".join(values)
                + f" | {_md_cell(winner)} | {_md_cell(notes)} |"
            )

    lines.extend(["", "## Unique Benchmark Coverage", ""])
    if not report.unique_tasks_per_paper and not report.unique_rows_per_paper:
        lines.append("_No unique benchmark coverage detected._")
    else:
        for item in report.papers_summary:
            paper_index = item["paper_index"]
            tasks = report.unique_tasks_per_paper.get(paper_index, [])
            rows = report.unique_rows_per_paper.get(paper_index, [])
            task_text = ", ".join(tasks) if tasks else "_none_"
            row_text = ", ".join(rows) if rows else "_none_"
            lines.append(f"- **Paper {paper_index} tasks:** {_md_cell(task_text)}")
            lines.append(f"- **Paper {paper_index} rows:** {_md_cell(row_text)}")

    lines.extend(["", "## Trade-offs", "", report.trade_offs])

    lines.extend(["", "## Recommendations", ""])
    if not report.recommendations:
        lines.append("_No constraint-specific recommendations available._")
    else:
        for rec in report.recommendations:
            lines.append(
                f"- **{_md_cell(rec.constraint)}:** Paper {rec.recommended_paper_index}. "
                f"{_md_cell(rec.reasoning)}"
            )

    lines.extend(["", "## Overall", ""])
    if report.overall_winner_index is None:
        lines.append("**Overall winner:** no clear winner")
    else:
        lines.append(f"**Overall winner:** Paper {report.overall_winner_index}")
    lines.append(f"**Winner basis:** `{report.winner_basis}`")
    if report.wins_by_paper:
        wins = ", ".join(
            f"Paper {paper_index}: {wins}"
            for paper_index, wins in sorted(report.wins_by_paper.items())
        )
        lines.append(f"**Benchmark row wins:** {wins}")
    lines.append("")
    lines.append(report.overall_winner_reasoning)

    return "\n".join(lines)


def comparator_agent(state: PaperIntelState | dict) -> dict:
    logger.info("Comparator agent started")

    papers = _as_paper_slots(state.get("papers", []))
    if len(papers) < 2:
        return {
            "comparison_report": None,
            "comparison_markdown": "",
            "processing_stage": state.get("processing_stage", "completed"),
            "errors": [_warning_error("Comparator skipped: fewer than two papers")],
        }

    papers_summary = _build_papers_summary(papers)
    matrix = _build_comparison_matrix(papers)
    unique_tasks = _build_unique_tasks_per_paper(papers, matrix)
    unique_rows = _build_unique_rows_per_paper(papers, matrix)
    matrix_stats = _build_matrix_stats(papers, matrix)

    evidence_json = _build_evidence_json(
        papers,
        matrix,
        unique_tasks,
        unique_rows,
        papers_summary,
        matrix_stats,
    )

    claims: dict = {}
    parsed = None
    parse_error = None
    repaired = None
    repair_error = None
    raw, llm_error = _call_llm(evidence_json)

    if llm_error:
        logger.warning("Comparator LLM unavailable, using deterministic fallback: %s", llm_error)
    else:
        parsed, parse_error = _parse_claims(raw or "")
        if parse_error:
            repaired, repair_error = _call_llm_repair(raw or "")
            if repair_error:
                logger.warning("Comparator repair failed, using deterministic fallback: %s", repair_error)
            else:
                parsed, parse_error = _parse_claims(repaired or "")
                if parse_error:
                    logger.warning("Comparator parse failed after repair, using fallback: %s", parse_error)

        if parsed is not None and not parse_error:
            claims = parsed

    report = _normalize_comparison_report(
        claims,
        papers,
        matrix,
        unique_tasks,
        unique_rows,
        papers_summary,
        matrix_stats,
    )
    markdown = _render_comparison_markdown(report)

    logger.info(
        "Comparator complete: papers=%d matrix_rows=%d winner=%s",
        len(papers),
        len(matrix),
        report.overall_winner_index,
    )

    return {
        "comparison_report": report,
        "comparison_markdown": markdown,
        "processing_stage": "comparison_completed",
    }
