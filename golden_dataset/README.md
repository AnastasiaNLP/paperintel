# PaperIntel Golden Dataset

This directory contains manually verified golden labels for PaperIntel
evaluation.

- `seed_5.jsonl`: small local seed used for fast development and CI-style
  contract checks.
- `paperintel_30_v0_1.jsonl`: full 30-paper evaluation dataset intended for
  portfolio and Hugging Face publication.
- `SCHEMA.md`: schema contract and labeling rules.
- `HF_DATASET_CARD.md`: Hugging Face dataset card template.
- `HF_PUBLISH.md`: Hugging Face publication checklist.

## Format

The JSONL files use one JSON object per line. Each row represents one paper and
mirrors the persisted `PaperWorkspace` artifact contract:

- `expected_method_extraction` maps to `MethodExtraction`
- `expected_benchmarks` maps to `list[BenchmarkResult]`
- `expected_readiness` maps to `ProductionReadiness`
- `expected_report_coverage` checks report text coverage
- `expected_report_judgment` configures later G-Eval checks for subjective
  report verdict fields
- `qa_cases` checks grounded QA behavior and citation coverage

## Loader Mapping Contract

Some golden fields are eval annotations rather than persisted model field names.
The loader must apply these mappings explicitly:

- `expected_method_extraction.description_keywords` checks substrings in
  `method_extraction_json["description"]`.
- `expected_method_extraction.novelty_keywords` checks substrings in
  `method_extraction_json["novelty_claim"]`.
- `expected_benchmarks[].conditions_keywords` checks substrings in the matched
  benchmark result's `conditions` string.
- `expected_readiness.expected_framework_integrations` checks
  `production_readiness_json["framework_integrations"]`.

## Labeling Rules

Labels must be based only on information present in the paper or links
explicitly provided by the paper. Do not use retrospective ecosystem knowledge,
later framework adoption, later Hugging Face implementations, or later
production usage as ground truth.

For benchmarks, use values from the main result tables. If the abstract and a
result table disagree, the result table wins. The 30-paper dataset keeps a
representative set of 4-8 benchmark rows per paper, prioritizing headline
results and important ablations without transcribing every table row.

`ProductionReadiness` is evaluated structurally. Do not put text
`must_mention` checks inside readiness. Text coverage belongs in
`expected_report_coverage`.

`recommended_action`, `implementation_difficulty`, and `action_reasoning` are
judgment-style report fields. They are not deterministic ground truth in this
dataset; evaluate them later with `expected_report_judgment`.

QA cases may check a narrower subset than the artifact benchmark list, but the
question wording should make that scope explicit.

## Validation

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/seed_5.jsonl
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

The 30-paper dataset should report `OK records=30`.
