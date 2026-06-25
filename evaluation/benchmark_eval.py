from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.golden_dataset import (
    GoldenBenchmarkV01,
    GoldenDatasetRecord,
    GoldenDatasetRecordV02,
)
from models.artifacts import PaperWorkspace


MATCHED = "matched"
TASK_MISMATCH = "task_mismatch"
METRIC_MISMATCH = "metric_mismatch"
VALUE_MISMATCH = "value_mismatch"
UNIT_MISMATCH = "unit_mismatch"
DATASET_MISMATCH = "dataset_mismatch"
CONDITION_MISMATCH = "condition_mismatch"
MISSING = "missing"


@dataclass(frozen=True)
class BenchmarkExpectedRow:
    task: str
    metric: str
    value: float
    unit: str | None
    conditions_keywords: list[str]
    dataset: str | None
    source_section: str | None
    source_table_or_figure: str | None
    reported_as: str | None
    value_type: str | None
    evidence_anchor: dict[str, Any] | None
    evidence_confidence: float | None
    paper_family: str | None
    difficulty_tags: list[str]
    label_quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "conditions_keywords": self.conditions_keywords,
            "dataset": self.dataset,
            "source_section": self.source_section,
            "source_table_or_figure": self.source_table_or_figure,
            "reported_as": self.reported_as,
            "value_type": self.value_type,
            "evidence_anchor": self.evidence_anchor,
            "evidence_confidence": self.evidence_confidence,
            "paper_family": self.paper_family,
            "difficulty_tags": self.difficulty_tags,
            "label_quality": self.label_quality,
        }


@dataclass(frozen=True)
class BenchmarkMatchDiagnostic:
    expected: BenchmarkExpectedRow
    matched: dict[str, Any] | None
    status: str
    reason: str
    near_matches: list[dict[str, Any]]
    component_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected.to_dict(),
            "matched": self.matched,
            "status": self.status,
            "reason": self.reason,
            "near_matches": self.near_matches,
            "component_scores": self.component_scores,
        }


@dataclass(frozen=True)
class BenchmarkEvalResult:
    paper_id: str
    score: float
    passed: bool
    expected_count: int
    matched_count: int
    missing_count: int
    diagnostics: list[BenchmarkMatchDiagnostic]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "score": self.score,
            "passed": self.passed,
            "expected_count": self.expected_count,
            "matched_count": self.matched_count,
            "missing_count": self.missing_count,
            "diagnostics": [
                diagnostic.to_dict()
                for diagnostic in self.diagnostics
            ],
        }


@dataclass(frozen=True)
class BenchmarkCandidateScore:
    actual: dict[str, Any]
    component_scores: dict[str, float]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual": self.actual,
            "component_scores": self.component_scores,
            "score": self.score,
        }


def score_benchmark_candidate(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> BenchmarkCandidateScore:
    scores = _component_scores(expected, actual)
    return BenchmarkCandidateScore(
        actual=actual,
        component_scores=scores,
        score=sum(scores.values()) / len(scores),
    )


def evaluate_benchmarks(
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace | dict[str, Any],
) -> BenchmarkEvalResult:
    payload = workspace.model_dump() if isinstance(workspace, PaperWorkspace) else workspace
    actual_rows = payload.get("benchmarks_json") or []
    diagnostics = [
        _evaluate_expected_row(record, expected, actual_rows)
        for expected in record.expected_benchmarks
    ]
    matched_count = sum(1 for diagnostic in diagnostics if diagnostic.status == MATCHED)
    expected_count = len(record.expected_benchmarks)
    score = matched_count / expected_count if expected_count else 0.0
    return BenchmarkEvalResult(
        paper_id=record.paper_id,
        score=score,
        passed=matched_count == expected_count,
        expected_count=expected_count,
        matched_count=matched_count,
        missing_count=expected_count - matched_count,
        diagnostics=diagnostics,
    )


def _evaluate_expected_row(
    record: GoldenDatasetRecord,
    expected: GoldenBenchmarkV01,
    actual_rows: list[dict[str, Any]],
) -> BenchmarkMatchDiagnostic:
    expected_row = _expected_row(record, expected)
    scored = [
        (score_benchmark_candidate(expected, actual).component_scores, actual)
        for actual in actual_rows
    ]
    full_matches = [
        (scores, actual)
        for scores, actual in scored
        if _is_full_match(scores)
    ]
    if full_matches:
        scores, actual = full_matches[0]
        return BenchmarkMatchDiagnostic(
            expected=expected_row,
            matched=actual,
            status=MATCHED,
            reason="task, metric, value, unit, and conditions matched",
            near_matches=[],
            component_scores=scores,
        )

    near_matches = _near_matches(scored)
    status = _failure_status(scored)
    primary_scores = _primary_component_scores(scored, status)
    return BenchmarkMatchDiagnostic(
        expected=expected_row,
        matched=None,
        status=status,
        reason=_failure_reason(status),
        near_matches=near_matches,
        component_scores=primary_scores,
    )


def _expected_row(
    record: GoldenDatasetRecord,
    expected: GoldenBenchmarkV01,
) -> BenchmarkExpectedRow:
    evidence_anchor = getattr(expected, "evidence_anchor", None)
    return BenchmarkExpectedRow(
        task=expected.task,
        metric=expected.metric,
        value=expected.value,
        unit=expected.unit,
        conditions_keywords=expected.conditions_keywords,
        dataset=getattr(expected, "dataset", None),
        source_section=getattr(expected, "source_section", None),
        source_table_or_figure=getattr(expected, "source_table_or_figure", None),
        reported_as=getattr(expected, "reported_as", None),
        value_type=getattr(expected, "value_type", None),
        evidence_anchor=evidence_anchor.model_dump() if evidence_anchor else None,
        evidence_confidence=getattr(expected, "evidence_confidence", None),
        paper_family=record.paper_family if isinstance(record, GoldenDatasetRecordV02) else None,
        difficulty_tags=record.difficulty_tags if isinstance(record, GoldenDatasetRecordV02) else [],
        label_quality=record.label_quality,
    )


def _component_scores(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> dict[str, float]:
    task_score = 1.0 if _benchmark_text_equal(actual.get("task"), expected.task) else 0.0
    metric_score = 1.0 if _benchmark_text_equal(actual.get("metric"), expected.metric) else 0.0
    value_score = 1.0 if _values_equal(actual.get("value"), expected.value) else 0.0
    unit_score = _unit_score(expected.unit, actual.get("unit"))
    dataset_score = _dataset_score(expected, actual)
    condition_score = _condition_score(expected, actual)
    return {
        "task": task_score,
        "metric": metric_score,
        "value": value_score,
        "unit": unit_score,
        "dataset": dataset_score,
        "conditions": condition_score,
    }


def _is_full_match(scores: dict[str, float]) -> bool:
    return all(score == 1.0 for score in scores.values())


def _near_matches(
    scored: list[tuple[dict[str, float], dict[str, Any]]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        scored,
        key=lambda item: sum(item[0].values()),
        reverse=True,
    )
    return [
        {
            "actual": actual,
            "component_scores": scores,
            "score": sum(scores.values()) / len(scores),
        }
        for scores, actual in ranked[:limit]
        if sum(scores.values()) > 0.0
    ]


def _failure_status(scored: list[tuple[dict[str, float], dict[str, Any]]]) -> str:
    if not scored:
        return MISSING
    task_candidates = [scores for scores, _actual in scored if scores["task"] == 1.0]
    if not task_candidates:
        return TASK_MISMATCH
    metric_candidates = [
        scores
        for scores in task_candidates
        if scores["metric"] == 1.0
    ]
    if not metric_candidates:
        return METRIC_MISMATCH
    value_candidates = [
        scores
        for scores in metric_candidates
        if scores["value"] == 1.0
    ]
    if not value_candidates:
        return VALUE_MISMATCH
    unit_candidates = [
        scores
        for scores in value_candidates
        if scores["unit"] == 1.0
    ]
    if not unit_candidates:
        return UNIT_MISMATCH
    dataset_candidates = [
        scores
        for scores in unit_candidates
        if scores["dataset"] == 1.0
    ]
    if not dataset_candidates:
        return DATASET_MISMATCH
    if not any(scores["conditions"] == 1.0 for scores in dataset_candidates):
        return CONDITION_MISMATCH
    return MISSING


def _primary_component_scores(
    scored: list[tuple[dict[str, float], dict[str, Any]]],
    status: str,
) -> dict[str, float]:
    if not scored:
        return {}
    if status == TASK_MISMATCH:
        return _best_scores(scored)
    candidates = scored
    for component in _components_before_status(status):
        candidates = [
            (scores, actual)
            for scores, actual in candidates
            if scores[component] == 1.0
        ]
    return _best_scores(candidates or scored)


def _components_before_status(status: str) -> tuple[str, ...]:
    return {
        METRIC_MISMATCH: ("task",),
        VALUE_MISMATCH: ("task", "metric"),
        UNIT_MISMATCH: ("task", "metric", "value"),
        DATASET_MISMATCH: ("task", "metric", "value", "unit"),
        CONDITION_MISMATCH: ("task", "metric", "value", "unit", "dataset"),
        MISSING: (),
    }.get(status, ())


def _best_scores(scored: list[tuple[dict[str, float], dict[str, Any]]]) -> dict[str, float]:
    scores, _actual = max(
        scored,
        key=lambda item: sum(item[0].values()),
    )
    return scores


def _failure_reason(status: str) -> str:
    return {
        TASK_MISMATCH: "no actual row matched the expected task",
        METRIC_MISMATCH: "task matched, but metric did not match",
        VALUE_MISMATCH: "task and metric matched, but numeric value did not match",
        UNIT_MISMATCH: "task, metric, and value matched, but unit did not match",
        DATASET_MISMATCH: "task, metric, value, and unit matched, but dataset did not match",
        CONDITION_MISMATCH: "task, metric, value, unit, and dataset matched, but conditions did not match",
        MISSING: "no usable actual benchmark row was found",
    }[status]


def _unit_score(expected_unit: str | None, actual_unit: Any) -> float:
    if expected_unit is None:
        return 1.0
    if actual_unit in (None, ""):
        return 1.0
    return 1.0 if _canonical_unit(actual_unit) == _canonical_unit(expected_unit) else 0.0


def _condition_score(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> float:
    if not expected.conditions_keywords:
        return 1.0
    search_text = _actual_benchmark_search_text(actual)
    matches = [
        keyword
        for keyword in expected.conditions_keywords
        if _benchmark_keyword_matches(keyword, search_text)
    ]
    return len(matches) / len(expected.conditions_keywords)


def _dataset_score(
    expected: GoldenBenchmarkV01,
    actual: dict[str, Any],
) -> float:
    expected_dataset = getattr(expected, "dataset", None)
    if expected_dataset in (None, ""):
        return 1.0
    actual_dataset = actual.get("dataset")
    if _benchmark_text_equal(actual_dataset, expected_dataset):
        return 1.0
    search_text = _actual_benchmark_search_text(actual)
    return 1.0 if _benchmark_keyword_matches(expected_dataset, search_text) else 0.0


def _actual_benchmark_search_text(actual: dict[str, Any]) -> str:
    values = [
        actual.get("task"),
        actual.get("metric"),
        actual.get("dataset"),
        actual.get("conditions"),
    ]
    conditions_keywords = actual.get("conditions_keywords")
    if isinstance(conditions_keywords, list):
        values.extend(conditions_keywords)
    elif conditions_keywords:
        values.append(conditions_keywords)
    return " ".join(_normalize_text(value) for value in values)


def _values_equal(actual: Any, expected: float, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def _unit_equal(actual: Any, expected: Any) -> bool:
    return _canonical_unit(actual) == _canonical_unit(expected)


def _canonical_unit(value: Any) -> str:
    normalized = _normalize_text(value)
    aliases = {
        "%": "percent",
        "percentage": "percent",
        "percent": "percent",
        "x": "x",
    }
    return aliases.get(normalized, normalized)


_BENCHMARK_TEXT_ALIASES = {
    "nq": "natural questions",
    "natural questions": "natural questions",
    "tqa": "triviaqa",
    "triviaqa": "triviaqa",
    "wq": "webquestions",
    "webquestions": "webquestions",
    "ct": "curatedtrec",
    "curated trec": "curatedtrec",
    "curatedtrec": "curatedtrec",
    "em": "exact match",
    "exact match": "exact match",
    "acc": "accuracy",
    "accuracy": "accuracy",
    "rag seq": "rag sequence",
    "rag seq.": "rag sequence",
    "rag sequence": "rag sequence",
}

_BENCHMARK_CONDITION_ALIASES = {
    "natural questions": {"natural questions", "nq"},
    "triviaqa": {"triviaqa", "tqa"},
    "webquestions": {"webquestions", "wq"},
    "curatedtrec": {"curatedtrec", "curated trec", "ct"},
    "exact match": {"exact match", "em"},
    "accuracy": {"accuracy", "acc"},
    "rag sequence": {"rag sequence", "rag seq", "rag seq."},
}


def _benchmark_text_equal(actual: Any, expected: Any) -> bool:
    return _canonical_benchmark_text(actual) == _canonical_benchmark_text(expected)


def _canonical_benchmark_text(value: Any) -> str:
    normalized = _normalize_text(value)
    return _BENCHMARK_TEXT_ALIASES.get(normalized, normalized)


def _benchmark_keyword_matches(keyword: str, normalized_actual: str) -> bool:
    normalized_keyword = _normalize_text(keyword)
    canonical_keyword = _canonical_benchmark_text(normalized_keyword)
    variants = {normalized_keyword, canonical_keyword}
    variants.update(_BENCHMARK_CONDITION_ALIASES.get(canonical_keyword, set()))
    return any(_contains_normalized_phrase(normalized_actual, variant) for variant in variants)


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    text_tokens = text.split()
    phrase_tokens = phrase.split()
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    return any(
        text_tokens[index : index + len(phrase_tokens)] == phrase_tokens
        for index in range(len(text_tokens) - len(phrase_tokens) + 1)
    )


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())
