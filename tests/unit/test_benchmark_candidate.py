import pytest
from pydantic import ValidationError

from models.schemas import BenchmarkCandidate


def test_benchmark_candidate_allows_partial_nullable_rows():
    candidate = BenchmarkCandidate()

    assert candidate.model_dump() == {
        "task": None,
        "dataset": None,
        "metric": None,
        "value": None,
        "unit": None,
        "method_or_model": None,
        "variant_role": "unknown",
        "benchmark_kind": "unknown",
        "source_section": None,
        "source_table_or_figure": None,
        "caption_context": None,
        "row_context": None,
        "conditions_keywords": [],
        "evidence_anchor": None,
        "evidence_confidence": None,
        "selection_status": "unknown",
        "selection_reason": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant_role", "primary"),
        ("benchmark_kind", "accuracy"),
        ("selection_status", "selected"),
    ],
)
def test_benchmark_candidate_rejects_unknown_controlled_values(field, value):
    with pytest.raises(ValidationError):
        BenchmarkCandidate(**{field: value})
