from evaluation.fixtures import build_perfect_workspace
from evaluation.golden_dataset import load_golden_records
from evaluation.method_eval import evaluate_method


def _transformer_record():
    return load_golden_records("golden_dataset/seed_5.jsonl")[0]


def test_method_eval_scores_perfect_workspace():
    record = _transformer_record()
    workspace = build_perfect_workspace(record)

    result = evaluate_method(record, workspace)

    assert result.passed
    assert result.score == 1.0
    assert result.fields["method_name"].passed
    assert result.fields["description_keywords"].missing == []
    assert result.fields["novelty_keywords"].missing == []
    assert result.to_dict()["fields"]["method_name"]["passed"] is True


def test_method_eval_reports_missing_description_keywords():
    record = _transformer_record()
    workspace = build_perfect_workspace(record)
    method_json = dict(workspace.method_extraction_json)
    method_json["description"] = "A neural sequence transduction model."
    workspace = workspace.model_copy(update={"method_extraction_json": method_json})

    result = evaluate_method(record, workspace)

    description = result.fields["description_keywords"]
    assert not result.passed
    assert description.score == 0.2
    assert description.matched == ["sequence transduction"]
    assert description.missing == [
        keyword
        for keyword in record.expected_method_extraction.description_keywords
        if keyword != "sequence transduction"
    ]


def test_method_eval_missing_novelty_keywords_affects_total_score():
    record = _transformer_record()
    workspace = build_perfect_workspace(record)
    method_json = dict(workspace.method_extraction_json)
    method_json["novelty_claim"] = "A Transformer model."
    workspace = workspace.model_copy(update={"method_extraction_json": method_json})

    result = evaluate_method(record, workspace)

    assert not result.passed
    assert result.fields["novelty_keywords"].score < 1.0
    assert result.score < 1.0


def test_method_eval_list_fields_include_matched_and_missing_diagnostics():
    record = _transformer_record()
    workspace = build_perfect_workspace(record)
    method_json = dict(workspace.method_extraction_json)
    expected_components = record.expected_method_extraction.key_components
    method_json["key_components"] = expected_components[:2]
    workspace = workspace.model_copy(update={"method_extraction_json": method_json})

    result = evaluate_method(record, workspace)

    components = result.fields["key_components"]
    assert components.matched == expected_components[:2]
    assert components.missing == expected_components[2:]
    assert components.score == len(expected_components[:2]) / len(expected_components)


def test_method_eval_accepts_raw_dict_workspace():
    record = _transformer_record()
    workspace = build_perfect_workspace(record).model_dump()

    result = evaluate_method(record, workspace)

    assert result.passed
    assert result.score == 1.0


def test_method_eval_works_with_v02_record():
    record = load_golden_records("golden_dataset/paperintel_60_v2_2.jsonl")[0]
    method = record.expected_method_extraction
    workspace = {
        "method_extraction_json": {
            "method_name": method.method_name,
            "description": " ".join(method.description_keywords),
            "novelty_claim": " ".join(method.novelty_keywords),
            "key_components": method.key_components,
            "compared_to": method.compared_to,
            "limitations_stated": method.limitations_stated,
        }
    }

    result = evaluate_method(record, workspace)

    assert result.paper_id == record.paper_id
    assert result.passed
    assert result.score == 1.0
