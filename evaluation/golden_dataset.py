from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

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


class GoldenEvidenceAnchor(StrictBaseModel):
    section: str = Field(min_length=1)
    table_or_figure: str | None = None
    page: int | None = None


class GoldenBenchmarkV01(StrictBaseModel):
    task: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    unit: str | None = None
    conditions_keywords: list[str] = Field(min_length=1)


class GoldenBenchmarkV02(GoldenBenchmarkV01):
    conditions_keywords: list[str] = Field(default_factory=list)
    dataset: str | None = None
    conditions: str = Field(min_length=1)
    source_section: str = Field(min_length=1)
    source_table_or_figure: str | None = None
    reported_as: Literal["main_table", "text", "leaderboard"]
    higher_is_better: bool
    value_type: Literal["absolute", "relative", "speedup", "memory", "latency"]
    evidence_anchor: GoldenEvidenceAnchor
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    anchor_source: str | None = None
    pass2_match_reason: str | None = None
    review_note: str | None = None


# Backward-compatible public name used by existing deterministic metric tests.
GoldenBenchmark = GoldenBenchmarkV01


class GoldenReadinessV01(StrictBaseModel):
    has_open_code: bool
    code_url: str | None = None
    huggingface_model: str | None = None
    expected_framework_integrations: list[str] = Field(default_factory=list)
    min_gpu_requirement: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    maturity_level: str = Field(min_length=1)
    allowed_maturity_levels: list[str] = Field(min_length=1)


class GoldenReadinessV02(GoldenReadinessV01):
    readiness_evidence_type: str = Field(min_length=1)
    evidence_anchors: list[GoldenEvidenceAnchor] = Field(default_factory=list)
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    evidence_notes: str = Field(min_length=1)


class GoldenReportCoverageV01(StrictBaseModel):
    must_mention: list[str] = Field(min_length=1)


class GoldenReportCoverageV02(StrictBaseModel):
    must_mention: list[str] = Field(default_factory=list)


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


class GoldenQACaseV01(StrictBaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer_keywords: list[str] = Field(min_length=1)
    required_citation_paper_ids: list[str] = Field(min_length=1)
    min_citations: int = Field(ge=1)


class GoldenQACaseV02(GoldenQACaseV01):
    question_type: Literal[
        "main_contribution",
        "mechanism",
        "limitation",
        "benchmark",
        "comparison",
        "implementation",
    ]
    evidence_anchors: list[GoldenEvidenceAnchor] = Field(min_length=1)
    must_not_claim: list[str] = Field(default_factory=list)
    anchor_source: str | None = None
    evidence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    must_not_claim_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    must_not_claim_source: str | None = None
    review_note: str | None = None


class GoldenDatasetRecordV01(StrictBaseModel):
    dataset_version: Literal["v0.1"]
    paper_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    split: str = Field(min_length=1)
    label_quality: Literal["manual_verified"]
    expected_method_extraction: GoldenMethodExtraction
    expected_benchmarks: list[GoldenBenchmarkV01] = Field(min_length=1)
    expected_readiness: GoldenReadinessV01
    expected_report_judgment: GoldenReportJudgment
    expected_report_coverage: GoldenReportCoverageV01
    qa_cases: list[GoldenQACaseV01] = Field(min_length=1)
    label_notes: str = Field(min_length=1)


class GoldenDatasetRecordV02(StrictBaseModel):
    dataset_version: Literal["v0.2"]
    schema_version: Literal["0.2"]
    paper_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    split: str = Field(min_length=1)
    label_quality: Literal["manual_verified", "draft_machine", "review_ready"]
    paper_family: str = Field(min_length=1)
    difficulty_tags: list[str] = Field(default_factory=list)
    quality_focus: list[str] = Field(default_factory=list)
    evaluation_scenarios: list[str] = Field(default_factory=list)
    expected_method_extraction: GoldenMethodExtraction
    expected_benchmarks: list[GoldenBenchmarkV02] = Field(min_length=1)
    expected_readiness: GoldenReadinessV02
    expected_report_judgment: GoldenReportJudgment
    expected_report_coverage: GoldenReportCoverageV02
    qa_cases: list[GoldenQACaseV02] = Field(min_length=1)
    label_notes: str = Field(min_length=1)
    v02_review_flags: list[str] | None = None
    v02_enrichment_method: str | None = None
    v02_pass2_applied: int | None = Field(default=None, ge=0)


GoldenDatasetRecord: TypeAlias = GoldenDatasetRecordV01 | GoldenDatasetRecordV02


@dataclass(frozen=True)
class GoldenDatasetSummary:
    records: int
    duplicates: int
    duplicate_paper_ids: list[str]
    benchmark_rows: int
    qa_cases: int
    dataset_versions: dict[str, int]
    schema_versions: dict[str, int]
    label_quality: dict[str, int]
    paper_families: dict[str, int]
    difficulty_tags: dict[str, int]

    @property
    def duplicate_count(self) -> int:
        return self.duplicates

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "duplicates": self.duplicates,
            "duplicate_paper_ids": self.duplicate_paper_ids,
            "benchmark_rows": self.benchmark_rows,
            "qa_cases": self.qa_cases,
            "dataset_versions": self.dataset_versions,
            "schema_versions": self.schema_versions,
            "label_quality": self.label_quality,
            "paper_families": self.paper_families,
            "difficulty_tags": self.difficulty_tags,
        }


@dataclass(frozen=True)
class GoldenDatasetValidation:
    path: Path
    records: list[GoldenDatasetRecord]
    summary: GoldenDatasetSummary

    @property
    def valid(self) -> bool:
        return self.summary.duplicates == 0


def load_golden_records(
    path: str | Path = DEFAULT_GOLDEN_DATASET_PATH,
) -> list[GoldenDatasetRecord]:
    validation = validate_golden_file(path)
    if validation.summary.duplicates:
        duplicate_list = ", ".join(validation.summary.duplicate_paper_ids)
        raise GoldenDatasetError(
            f"Duplicate paper_id values in {validation.path}: {duplicate_list}"
        )
    return validation.records


def validate_golden_file(
    path: str | Path = DEFAULT_GOLDEN_DATASET_PATH,
) -> GoldenDatasetValidation:
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

    return GoldenDatasetValidation(
        path=dataset_path,
        records=records,
        summary=build_golden_summary(records),
    )


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

    version = payload.get("dataset_version")
    try:
        if version == "v0.1":
            return GoldenDatasetRecordV01.model_validate(payload)
        if version == "v0.2":
            return GoldenDatasetRecordV02.model_validate(payload)
    except ValidationError as exc:
        raise GoldenDatasetError(
            f"Invalid golden record in {path} at line {line_number}: {exc}"
        ) from exc

    raise GoldenDatasetError(
        f"Unsupported dataset_version in {path} at line {line_number}: {version!r}"
    )


def build_golden_summary(
    records: list[GoldenDatasetRecord],
) -> GoldenDatasetSummary:
    paper_ids = [record.paper_id for record in records]
    paper_id_counts = Counter(paper_ids)
    duplicate_paper_ids = sorted(
        paper_id for paper_id, count in paper_id_counts.items() if count > 1
    )
    dataset_versions = Counter(record.dataset_version for record in records)
    schema_versions = Counter(
        getattr(record, "schema_version", "none") for record in records
    )
    label_quality = Counter(record.label_quality for record in records)
    paper_families = Counter(
        record.paper_family
        for record in records
        if isinstance(record, GoldenDatasetRecordV02)
    )
    difficulty_tags: Counter[str] = Counter()
    for record in records:
        if isinstance(record, GoldenDatasetRecordV02):
            difficulty_tags.update(record.difficulty_tags)

    return GoldenDatasetSummary(
        records=len(records),
        duplicates=sum(paper_id_counts[paper_id] - 1 for paper_id in duplicate_paper_ids),
        duplicate_paper_ids=duplicate_paper_ids,
        benchmark_rows=sum(len(record.expected_benchmarks) for record in records),
        qa_cases=sum(len(record.qa_cases) for record in records),
        dataset_versions=dict(sorted(dataset_versions.items())),
        schema_versions=dict(sorted(schema_versions.items())),
        label_quality=dict(sorted(label_quality.items())),
        paper_families=dict(sorted(paper_families.items())),
        difficulty_tags=dict(sorted(difficulty_tags.items())),
    )


def summarize_golden_records(records: list[GoldenDatasetRecord]) -> str:
    paper_ids = ",".join(record.paper_id for record in records)
    return f"OK records={len(records)} paper_ids={paper_ids}"


def summarize_golden_validation(validation: GoldenDatasetValidation) -> str:
    summary = validation.summary
    lines = [
        f"OK records={summary.records}",
        f"duplicates={summary.duplicates}",
        f"benchmark_rows={summary.benchmark_rows}",
        f"qa_cases={summary.qa_cases}",
        f"dataset_versions={summary.dataset_versions}",
        f"schema_versions={summary.schema_versions}",
        f"label_quality={summary.label_quality}",
    ]
    if summary.paper_families:
        lines.append(f"paper_families={summary.paper_families}")
    if summary.difficulty_tags:
        lines.append(f"difficulty_tags={summary.difficulty_tags}")
    if summary.duplicate_paper_ids:
        lines.append(f"duplicate_paper_ids={summary.duplicate_paper_ids}")
    return "\n".join(lines)
