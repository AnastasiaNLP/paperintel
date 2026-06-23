import json

import pytest

from evaluation.golden_dataset import (
    GoldenDatasetError,
    GoldenDatasetRecordV02,
    load_golden_records,
    validate_golden_file,
)


V02_DATASET = "golden_dataset/paperintel_60_v2_2.jsonl"


def test_v02_dataset_loads_with_expected_contract_summary():
    validation = validate_golden_file(V02_DATASET)
    summary = validation.summary

    assert summary.records == 60
    assert summary.duplicates == 0
    assert summary.duplicate_paper_ids == []
    assert summary.benchmark_rows == 333
    assert summary.qa_cases == 180
    assert summary.dataset_versions == {"v0.2": 60}
    assert summary.schema_versions == {"0.2": 60}
    assert summary.label_quality == {
        "draft_machine": 30,
        "manual_verified": 30,
    }


def test_v02_dataset_records_have_required_metadata():
    records = load_golden_records(V02_DATASET)

    assert len(records) == 60
    assert all(isinstance(record, GoldenDatasetRecordV02) for record in records)
    assert all(record.dataset_version == "v0.2" for record in records)
    assert all(record.schema_version == "0.2" for record in records)
    assert all(record.paper_family for record in records)
    assert all(isinstance(record.difficulty_tags, list) for record in records)
    assert all(
        qa_case.question_type
        for record in records
        for qa_case in record.qa_cases
    )


def test_v02_dataset_summary_tracks_families_and_difficulty_tags():
    summary = validate_golden_file(V02_DATASET).summary

    assert summary.paper_families["architecture"] == 13
    assert summary.paper_families["agents"] == 12
    assert summary.difficulty_tags["table_heavy"] == 52
    assert summary.difficulty_tags["multi_dataset"] == 39


def test_existing_v01_datasets_still_load():
    seed_records = load_golden_records("golden_dataset/seed_5.jsonl")
    thirty_records = load_golden_records("golden_dataset/paperintel_30_v0_1.jsonl")

    assert len(seed_records) == 5
    assert len(thirty_records) == 30
    assert {record.dataset_version for record in seed_records} == {"v0.1"}
    assert {record.dataset_version for record in thirty_records} == {"v0.1"}


def test_v02_unknown_extra_field_is_rejected(tmp_path):
    record = load_golden_records(V02_DATASET)[0].model_dump()
    record["unexpected_field"] = "must fail"

    dataset_path = tmp_path / "extra.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(GoldenDatasetError, match="unexpected_field"):
        load_golden_records(dataset_path)


def test_v02_nested_benchmark_extra_field_is_rejected(tmp_path):
    record = load_golden_records(V02_DATASET)[0].model_dump()
    record["expected_benchmarks"][0]["unexpected_benchmark_field"] = "must fail"

    dataset_path = tmp_path / "benchmark-extra.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(GoldenDatasetError, match="unexpected_benchmark_field"):
        load_golden_records(dataset_path)


def test_v02_nested_qa_extra_field_is_rejected(tmp_path):
    record = load_golden_records(V02_DATASET)[0].model_dump()
    record["qa_cases"][0]["unexpected_qa_field"] = "must fail"

    dataset_path = tmp_path / "qa-extra.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(GoldenDatasetError, match="unexpected_qa_field"):
        load_golden_records(dataset_path)


def test_validation_collects_duplicate_diagnostics_before_load_failure(tmp_path):
    record = load_golden_records(V02_DATASET)[0].model_dump()
    dataset_path = tmp_path / "duplicates.jsonl"
    dataset_path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    validation = validate_golden_file(dataset_path)

    assert validation.summary.records == 2
    assert validation.summary.duplicates == 1
    assert validation.summary.duplicate_paper_ids == [record["paper_id"]]
    with pytest.raises(GoldenDatasetError, match="Duplicate paper_id"):
        load_golden_records(dataset_path)
