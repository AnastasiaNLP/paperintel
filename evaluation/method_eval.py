from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.golden_dataset import GoldenDatasetRecord
from models.artifacts import PaperWorkspace


@dataclass(frozen=True)
class MethodFieldScore:
    field: str
    score: float
    expected: list[str] | str
    matched: list[str]
    missing: list[str]

    @property
    def passed(self) -> bool:
        return self.score == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "score": self.score,
            "passed": self.passed,
            "expected": self.expected,
            "matched": self.matched,
            "missing": self.missing,
        }


@dataclass(frozen=True)
class MethodEvalResult:
    paper_id: str
    score: float
    passed: bool
    fields: dict[str, MethodFieldScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "score": self.score,
            "passed": self.passed,
            "fields": {
                name: score.to_dict()
                for name, score in self.fields.items()
            },
        }


def evaluate_method(
    record: GoldenDatasetRecord,
    workspace: PaperWorkspace | dict[str, Any],
) -> MethodEvalResult:
    payload = workspace.model_dump() if isinstance(workspace, PaperWorkspace) else workspace
    method_json = payload.get("method_extraction_json") or {}
    expected = record.expected_method_extraction

    fields = {
        "method_name": _text_field_score(
            field="method_name",
            expected=expected.method_name,
            actual=method_json.get("method_name"),
        ),
        "description_keywords": _text_keywords_score(
            field="description_keywords",
            expected=expected.description_keywords,
            actual=method_json.get("description"),
        ),
        "novelty_keywords": _text_keywords_score(
            field="novelty_keywords",
            expected=expected.novelty_keywords,
            actual=method_json.get("novelty_claim"),
        ),
        "key_components": _list_items_score(
            field="key_components",
            expected=expected.key_components,
            actual=method_json.get("key_components") or [],
        ),
        "compared_to": _list_items_score(
            field="compared_to",
            expected=expected.compared_to,
            actual=method_json.get("compared_to") or [],
        ),
        "limitations_stated": _list_items_score(
            field="limitations_stated",
            expected=expected.limitations_stated,
            actual=method_json.get("limitations_stated") or [],
        ),
    }
    score = _average([field.score for field in fields.values()])
    return MethodEvalResult(
        paper_id=record.paper_id,
        score=score,
        passed=score == 1.0,
        fields=fields,
    )


def _text_field_score(
    *,
    field: str,
    expected: str,
    actual: Any,
) -> MethodFieldScore:
    matched = [expected] if _contains_normalized(actual, expected) else []
    missing = [] if matched else [expected]
    return MethodFieldScore(
        field=field,
        score=1.0 if matched else 0.0,
        expected=expected,
        matched=matched,
        missing=missing,
    )


def _text_keywords_score(
    *,
    field: str,
    expected: list[str],
    actual: Any,
) -> MethodFieldScore:
    return _sequence_score(field=field, expected=expected, actual_text=actual)


def _list_items_score(
    *,
    field: str,
    expected: list[str],
    actual: list[Any],
) -> MethodFieldScore:
    actual_text = " ".join(_normalize_text(item) for item in actual)
    return _sequence_score(field=field, expected=expected, actual_text=actual_text)


def _sequence_score(
    *,
    field: str,
    expected: list[str],
    actual_text: Any,
) -> MethodFieldScore:
    if not expected:
        return MethodFieldScore(
            field=field,
            score=1.0,
            expected=[],
            matched=[],
            missing=[],
        )
    matched = [
        item
        for item in expected
        if _contains_normalized(actual_text, item)
    ]
    missing = [
        item
        for item in expected
        if item not in matched
    ]
    return MethodFieldScore(
        field=field,
        score=len(matched) / len(expected),
        expected=expected,
        matched=matched,
        missing=missing,
    )


def _contains_normalized(actual: Any, expected: str) -> bool:
    normalized_actual = _normalize_text(actual)
    normalized_expected = _normalize_text(expected)
    return bool(normalized_expected and normalized_expected in normalized_actual)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
