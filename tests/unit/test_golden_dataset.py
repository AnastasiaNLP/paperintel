import json

import pytest

from evaluation.golden_dataset import (
    LOADER_FIELD_MAPPINGS,
    GoldenDatasetError,
    load_golden_records,
    summarize_golden_records,
)


EXPECTED_SEED_IDS = [
    "1706.03762",
    "2005.11401",
    "2106.09685",
    "2210.03629",
    "2205.14135",
]


def test_load_golden_seed_records():
    records = load_golden_records("golden_dataset/seed_5.jsonl")

    assert [record.paper_id for record in records] == EXPECTED_SEED_IDS
    assert summarize_golden_records(records) == (
        "OK records=5 paper_ids="
        "1706.03762,2005.11401,2106.09685,2210.03629,2205.14135"
    )


def test_golden_seed_uses_real_readiness_field_mapping():
    records = load_golden_records("golden_dataset/seed_5.jsonl")

    readiness = records[0].expected_readiness

    assert readiness.expected_framework_integrations == ["TensorFlow"]
    assert not hasattr(readiness, "expected_frameworks")


def test_loader_field_mappings_document_eval_annotations():
    assert LOADER_FIELD_MAPPINGS == {
        "expected_method_extraction.description_keywords": (
            "method_extraction_json.description"
        ),
        "expected_method_extraction.novelty_keywords": (
            "method_extraction_json.novelty_claim"
        ),
        "expected_benchmarks[].conditions_keywords": "benchmarks_json[].conditions",
        "expected_readiness.expected_framework_integrations": (
            "readiness_json.framework_integrations"
        ),
    }


def test_legacy_expected_frameworks_key_is_rejected(tmp_path):
    record = load_golden_records("golden_dataset/seed_5.jsonl")[0].model_dump()
    readiness = record["expected_readiness"]
    readiness["expected_frameworks"] = readiness.pop("expected_framework_integrations")

    dataset_path = tmp_path / "legacy.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(GoldenDatasetError, match="expected_framework"):
        load_golden_records(dataset_path)

