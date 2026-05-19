from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.artifacts import PaperWorkspace

from evaluation.golden_dataset import GoldenBenchmark, GoldenDatasetRecord


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    score: float
    details: dict[str, Any]


@dataclass(frozen=True)
class WorkspaceEvaluation:
    paper_id: str
    checks: list[CheckResult]

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(check.score for check in self.checks) / len(self.checks)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def evaluate_workspace(
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace | dict[str, Any],
) -> WorkspaceEvaluation:
    payload = workspace.model_dump() if isinstance(workspace, PaperWorkspace) else workspace
    checks = [
        evaluate_method_extraction(
            record,
            payload.get("method_extraction_json") or {},
        ),
        evaluate_benchmarks(
            record,
            payload.get("benchmarks_json") or [],
        ),
        evaluate_readiness(
            record,
            payload.get("readiness_json") or {},
        ),
        evaluate_report_coverage(
            record,
            payload.get("full_markdown_report") or "",
        ),
    ]
    return WorkspaceEvaluation(paper_id=record.paper_id, checks=checks)


def evaluate_method_extraction(
    record: GoldenDatasetRecord,
    method_json: dict[str, Any],
) -> CheckResult:
    expected = record.expected_method_extraction
    method_name_score = _field_contains(
        method_json.get("method_name"),
        expected.method_name,
    )
    description_score = _keyword_coverage(
        expected.description_keywords,
        method_json.get("description"),
    )
    novelty_score = _keyword_coverage(
        expected.novelty_keywords,
        method_json.get("novelty_claim"),
    )
    components_score = _list_coverage(
        expected.key_components,
        method_json.get("key_components") or [],
    )
    compared_score = _list_coverage(
        expected.compared_to,
        method_json.get("compared_to") or [],
    )
    limitations_score = _list_coverage(
        expected.limitations_stated,
        method_json.get("limitations_stated") or [],
    )
    scores = [
        method_name_score,
        description_score,
        novelty_score,
        components_score,
        compared_score,
        limitations_score,
    ]
    score = _average(scores)
    return CheckResult(
        name="method_extraction",
        passed=score == 1.0,
        score=score,
        details={
            "method_name": method_name_score,
            "description_keywords": description_score,
            "novelty_keywords": novelty_score,
            "key_components": components_score,
            "compared_to": compared_score,
            "limitations_stated": limitations_score,
        },
    )


def evaluate_benchmarks(
    record: GoldenDatasetRecord,
    benchmarks_json: list[dict[str, Any]],
) -> CheckResult:
    matched = [
        expected
        for expected in record.expected_benchmarks
        if _find_matching_benchmark(expected, benchmarks_json) is not None
    ]
    score = len(matched) / len(record.expected_benchmarks)
    return CheckResult(
        name="benchmarks",
        passed=score == 1.0,
        score=score,
        details={
            "expected_count": len(record.expected_benchmarks),
            "matched_count": len(matched),
            "missing": [
                _benchmark_key(expected)
                for expected in record.expected_benchmarks
                if expected not in matched
            ],
        },
    )


def evaluate_readiness(
    record: GoldenDatasetRecord,
    readiness_json: dict[str, Any],
) -> CheckResult:
    expected = record.expected_readiness
    checks = {
        "has_open_code": readiness_json.get("has_open_code") == expected.has_open_code,
        "code_url": _optional_exact(readiness_json.get("code_url"), expected.code_url),
        "huggingface_model": _optional_exact(
            readiness_json.get("huggingface_model"),
            expected.huggingface_model,
        ),
        "framework_integrations": _list_coverage(
            expected.expected_framework_integrations,
            readiness_json.get("framework_integrations") or [],
        )
        == 1.0,
        "min_gpu_requirement": _optional_exact(
            readiness_json.get("min_gpu_requirement"),
            expected.min_gpu_requirement,
        ),
        "dependencies": _list_coverage(
            expected.dependencies,
            readiness_json.get("dependencies") or [],
        )
        == 1.0,
        "maturity_level": readiness_json.get("maturity_level")
        in expected.allowed_maturity_levels,
    }
    score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    return CheckResult(
        name="readiness",
        passed=score == 1.0,
        score=score,
        details=checks,
    )


def evaluate_report_coverage(
    record: GoldenDatasetRecord,
    report_text: str,
) -> CheckResult:
    score = _keyword_coverage(
        record.expected_report_coverage.must_mention,
        report_text,
    )
    return CheckResult(
        name="report_coverage",
        passed=score == 1.0,
        score=score,
        details={
            "must_mention": record.expected_report_coverage.must_mention,
        },
    )


def _find_matching_benchmark(
    expected: GoldenBenchmark,
    actual_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for actual in actual_rows:
        if not _normalized_equal(actual.get("task"), expected.task):
            continue
        if not _normalized_equal(actual.get("metric"), expected.metric):
            continue
        if not _values_equal(actual.get("value"), expected.value):
            continue
        if expected.unit is not None and not _normalized_equal(
            actual.get("unit"),
            expected.unit,
        ):
            continue
        if _keyword_coverage(
            expected.conditions_keywords,
            actual.get("conditions"),
        ) != 1.0:
            continue
        return actual
    return None


def _benchmark_key(benchmark: GoldenBenchmark) -> str:
    return f"{benchmark.task} | {benchmark.metric} | {benchmark.value}"


def _optional_exact(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual in (None, "")
    return _normalized_equal(actual, expected)


def _field_contains(actual: Any, expected: str) -> float:
    actual_text = _normalize_text(actual)
    expected_text = _normalize_text(expected)
    return 1.0 if expected_text and expected_text in actual_text else 0.0


def _keyword_coverage(expected_keywords: list[str], actual_text: Any) -> float:
    if not expected_keywords:
        return 1.0
    normalized_actual = _normalize_text(actual_text)
    matches = [
        keyword
        for keyword in expected_keywords
        if _normalize_text(keyword) in normalized_actual
    ]
    return len(matches) / len(expected_keywords)


def _list_coverage(expected_items: list[str], actual_items: list[Any]) -> float:
    if not expected_items:
        return 1.0
    actual_text = " ".join(_normalize_text(item) for item in actual_items)
    matches = [
        item
        for item in expected_items
        if _normalize_text(item) in actual_text
    ]
    return len(matches) / len(expected_items)


def _normalized_equal(actual: Any, expected: Any) -> bool:
    return _normalize_text(actual) == _normalize_text(expected)


def _values_equal(actual: Any, expected: float, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
