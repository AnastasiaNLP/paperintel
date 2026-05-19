from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


DEFAULT_GOLDEN_DATASET_PATH = Path("golden_dataset/seed_5.jsonl")

EXPECTED_JUDGMENT_FIELDS = {
    "recommended_action",
    "implementation_difficulty",
    "action_reasoning",
}

LOADER_FIELD_MAPPINGS = {
    "expected_method_extraction.description_keywords": "method_extraction_json.description",
    "expected_method_extraction.novelty_keywords": "method_extraction_json.novelty_claim",
    "expected_benchmarks[].conditions_keywords": "benchmarks_json[].conditions",
    "expected_readiness.expected_framework_integrations": (
        "readiness_json.framework_integrations"
    ),
}


class GoldenDatasetError(ValueError):
    """Raised when a golden dataset file cannot be loaded or validated."""


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldenMethodExtraction(StrictBaseModel):
    method_name: str = Field(min_length=1)
    description_keywords: list[str] = Field(min_length=1)
    novelty_keywords: list[str] = Field(min_length=1)
    key_components: list[str] = Field(default_factory=list)
    compared_to: list[str] = Field(default_factory=list)
    limitations_stated: list[str] = Field(default_factory=list)


class GoldenBenchmark(StrictBaseModel):
    task: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    unit: str | None = None
    conditions_keywords: list[str] = Field(min_length=1)


class GoldenReadiness(StrictBaseModel):
    has_open_code: bool
    code_url: str | None = None
    huggingface_model: str | None = None
    expected_framework_integrations: list[str] = Field(default_factory=list)
    min_gpu_requirement: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    maturity_level: str = Field(min_length=1)
    allowed_maturity_levels: list[str] = Field(min_length=1)


class GoldenReportCoverage(StrictBaseModel):
    must_mention: list[str] = Field(min_length=1)


class GoldenReportJudgment(StrictBaseModel):
    eval_mode: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)

    @field_validator("eval_mode")
    @classmethod
    def require_g_eval(cls, value: str) -> str:
        if value != "g_eval":
            raise ValueError("expected_report_judgment.eval_mode must be 'g_eval'")
        return value

    @field_validator("fields")
    @classmethod
    def require_expected_judgment_fields(cls, value: list[str]) -> list[str]:
        if set(value) != EXPECTED_JUDGMENT_FIELDS:
            raise ValueError(
                "expected_report_judgment.fields must contain "
                f"{sorted(EXPECTED_JUDGMENT_FIELDS)}"
            )
        return value


class GoldenQACase(StrictBaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer_keywords: list[str] = Field(min_length=1)
    required_citation_paper_ids: list[str] = Field(min_length=1)
    min_citations: int = Field(ge=1)


class GoldenDatasetRecord(StrictBaseModel):
    dataset_version: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    split: str = Field(min_length=1)
    label_quality: str = Field(min_length=1)
    expected_method_extraction: GoldenMethodExtraction
    expected_benchmarks: list[GoldenBenchmark] = Field(min_length=1)
    expected_readiness: GoldenReadiness
    expected_report_judgment: GoldenReportJudgment
    expected_report_coverage: GoldenReportCoverage
    qa_cases: list[GoldenQACase] = Field(min_length=1)
    label_notes: str = Field(min_length=1)


def load_golden_records(
    path: str | Path = DEFAULT_GOLDEN_DATASET_PATH,
) -> list[GoldenDatasetRecord]:
    dataset_path = Path(path)
    records: list[GoldenDatasetRecord] = []

    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GoldenDatasetError(f"Could not read golden dataset: {dataset_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records.append(_parse_record(line, line_number=line_number, path=dataset_path))

    if not records:
        raise GoldenDatasetError(f"Golden dataset is empty: {dataset_path}")

    _validate_unique_paper_ids(records, path=dataset_path)
    return records


def _parse_record(
    line: str,
    *,
    line_number: int,
    path: Path,
) -> GoldenDatasetRecord:
    try:
        payload: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(
            f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
        ) from exc

    try:
        return GoldenDatasetRecord.model_validate(payload)
    except ValidationError as exc:
        raise GoldenDatasetError(
            f"Invalid golden record in {path} at line {line_number}: {exc}"
        ) from exc


def _validate_unique_paper_ids(
    records: list[GoldenDatasetRecord],
    *,
    path: Path,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.paper_id in seen:
            duplicates.append(record.paper_id)
        seen.add(record.paper_id)

    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise GoldenDatasetError(
            f"Duplicate paper_id values in {path}: {duplicate_list}"
        )


def summarize_golden_records(records: list[GoldenDatasetRecord]) -> str:
    paper_ids = ",".join(record.paper_id for record in records)
    return f"OK records={len(records)} paper_ids={paper_ids}"

