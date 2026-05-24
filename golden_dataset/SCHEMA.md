# PaperIntel Golden Dataset Schema v0.1

This schema describes the JSONL records used by PaperIntel evaluation. Each line
is one manually verified paper record.

## Files

- `seed_5.jsonl`: small local development and CI seed.
- `paperintel_30_v0_1.jsonl`: full 30-paper manually verified evaluation
  dataset intended for portfolio and Hugging Face publication.

## Record Shape

Required top-level fields:

- `dataset_version`: currently `v0.1`.
- `paper_id`: arXiv ID without prefix, for example `1706.03762`.
- `source_url`: canonical arXiv abstract URL.
- `title`: paper title.
- `domain`: snake_case topic label.
- `split`: currently `eval`.
- `label_quality`: currently `manual_verified`.
- `expected_method_extraction`: method extraction labels.
- `expected_benchmarks`: benchmark result labels.
- `expected_readiness`: production-readiness labels.
- `expected_report_judgment`: rubric configuration for later LLM-judge scoring.
- `expected_report_coverage`: keyword coverage requirements for the report.
- `qa_cases`: grounded QA cases and citation expectations.
- `label_notes`: reviewer notes about labeling decisions.

## Method Extraction

`expected_method_extraction` mirrors the `MethodExtraction` artifact, with two
eval-specific keyword fields:

- `method_name`: canonical method or system name.
- `description_keywords`: substrings expected in the extracted description.
- `novelty_keywords`: substrings expected in the extracted novelty claim.
- `key_components`: architecture, algorithm, or system components.
- `compared_to`: explicit baselines or related methods.
- `limitations_stated`: limitations stated by the authors; use `[]` when none
  are explicit.

Loader mapping:

- `description_keywords` checks `method_extraction_json["description"]`.
- `novelty_keywords` checks `method_extraction_json["novelty_claim"]`.

## Benchmarks

`expected_benchmarks` is a list of representative benchmark rows. Each row has:

- `task`: task or benchmark family.
- `metric`: metric name.
- `value`: numeric value as a JSON number.
- `unit`: optional unit such as `percent`, `x`, or `pairs`; otherwise `null`.
- `conditions_keywords`: at least two substrings identifying the model, dataset,
  split, setting, or ablation row.

Benchmark labels should come from main result tables, not abstracts. If an
abstract and table disagree, the table value wins. The full 30-paper dataset
keeps 4-8 rows per paper so deterministic recall remains useful without turning
the dataset into a full table transcription.

## Readiness

`expected_readiness` mirrors the `ProductionReadiness` artifact:

- `has_open_code`
- `code_url`
- `huggingface_model`
- `expected_framework_integrations`
- `min_gpu_requirement`
- `dependencies`
- `maturity_level`
- `allowed_maturity_levels`

Allowed `maturity_level` values:

- `research_only`
- `experimental`
- `production_ready`

Readiness labels must use only evidence present in the paper or links explicitly
provided by the paper. Do not use later ecosystem adoption, later Hugging Face
ports, or later production usage as ground truth.

## Report Coverage And Judgment

`expected_report_coverage.must_mention` contains 5-8 central concepts that
should appear in the engineer-facing report. This is a coverage proxy, not a
semantic correctness metric.

`expected_report_judgment` is fixed for v0.1:

```json
{
  "eval_mode": "g_eval",
  "fields": [
    "recommended_action",
    "implementation_difficulty",
    "action_reasoning"
  ]
}
```

These fields are subjective and are not deterministic ground truth. They are
reserved for the LLM-judge rubric layer.

## QA Cases

Each record has exactly three QA cases:

- one main contribution question;
- one method, architecture, or mechanism question;
- one headline result, limitation, or operational implication question.

Each QA case includes:

- `id`
- `question`
- `expected_answer_keywords`
- `required_citation_paper_ids`
- `min_citations`

For v0.1, `required_citation_paper_ids` should be `[paper_id]` and
`min_citations` should be `1`.

## Validation

Run:

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

Expected output starts with:

```text
OK records=30
```

