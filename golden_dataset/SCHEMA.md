# PaperIntel Golden Dataset Schema

This document is the public contract for PaperIntel golden dataset JSONL files.
Each line is one paper record. The loader must be strict: unknown fields are
schema errors, not warnings.

## Files

- `seed_5.jsonl`: small local development and CI seed.
- `paperintel_30_v0_1.jsonl`: original 30-paper manually verified evaluation
  dataset.
- `paperintel_60_v2_2.jsonl`: current 60-paper v0.2 `review_ready` dataset.

The current 60-paper filename is historical. Its records declare:

```text
dataset_version = "v0.2"
schema_version = "0.2"
```

Those record fields are the source of truth for dataset and schema versioning.

## Version Policy

The evaluation package supports versioned record models. Do not make the loader
permissive to bridge versions. Add exact models for new schema versions while
keeping `extra="forbid"`.

Supported records:

- v0.1: `dataset_version="v0.1"` and no `schema_version` field.
- v0.2: `dataset_version="v0.2"` and `schema_version="0.2"`.

Summaries and evaluation results must report version breakdowns when multiple
files or versions are involved.

## Label Quality

`label_quality` controls how strongly a record can be used for regression
decisions.

- `manual_verified`: reviewer-checked labels. These records can be used for
  strict deterministic gates.
- `draft_machine`: machine-generated draft labels. These records are useful for
  exploratory coverage, diagnostics, and trend tracking, but should not be mixed
  into hard gates without a separate threshold or explicit lower-confidence
  policy.
- `review_ready`: labels prepared for review. Treat as non-gating unless manual
  verification has been recorded.

Evaluation summaries and runners must include breakdowns by `label_quality`.

## v0.1 Record Shape

Required top-level fields:

- `dataset_version`
- `paper_id`
- `source_url`
- `title`
- `domain`
- `split`
- `label_quality`
- `expected_method_extraction`
- `expected_benchmarks`
- `expected_readiness`
- `expected_report_judgment`
- `expected_report_coverage`
- `qa_cases`
- `label_notes`

For v0.1, `dataset_version` is `v0.1` and `label_quality` is
`manual_verified`.

## v0.2 Record Shape

Required top-level fields:

- all v0.1 fields
- `schema_version`
- `paper_family`
- `difficulty_tags`
- `quality_focus`
- `evaluation_scenarios`

Optional top-level review metadata:

- `v02_review_flags`
- `v02_enrichment_method`
- `v02_pass2_applied`: integer count of second-pass updates applied to the
  record.

For `paperintel_60_v2_2.jsonl`, every record must have:

```text
dataset_version = "v0.2"
schema_version = "0.2"
```

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

`expected_benchmarks` is a list of representative benchmark rows.

Required v0.1 fields:

- `task`: task or benchmark family.
- `metric`: metric name.
- `value`: numeric value as a JSON number.
- `unit`: optional unit such as `percent`, `x`, `GB`, `tokens/s`, or `pairs`;
  otherwise `null`.
- `conditions_keywords`: substrings identifying the model, dataset, split,
  setting, or ablation row.

Additional required v0.2 fields:

- `dataset`: benchmark dataset or evaluated system context; may be `null` when
  the result is not tied to a named dataset.
- `conditions`: plain-text row context.
- `source_section`: paper section where the result appears.
- `source_table_or_figure`: table or figure identifier, or `null`.
- `reported_as`: source category used by the dataset.
- `higher_is_better`: whether larger values are better for this metric.
- `value_type`: stable value class.
- `evidence_anchor`: source pointer for the row.
- `evidence_confidence`: numeric confidence in `[0.0, 1.0]`.

For v0.2, `conditions_keywords` may be empty when the row is already identified
by task, metric, value, dataset, and source metadata.

Current `reported_as` values in `paperintel_60_v2_2.jsonl`:

- `main_table`
- `text`
- `leaderboard`

Current `value_type` values in `paperintel_60_v2_2.jsonl`:

- `absolute`
- `relative`
- `speedup`
- `memory`
- `latency`

Treat these as controlled vocabularies for v0.2. Add new values only with an
explicit schema update.

Optional v0.2 review fields:

- `anchor_source`
- `pass2_match_reason`
- `review_note`

Benchmark labels should come from result tables or explicitly reported result
statements. If an abstract and result table disagree, the table value wins. The
dataset keeps representative rows so deterministic recall remains useful
without turning each paper into a full table transcription.

## Evidence Anchors

Evidence anchors identify where a claim or expected label is supported in the
paper. They are used for deterministic grounding checks before any LLM judge
score is considered.

Shape:

```json
{
  "section": "Experiments",
  "table_or_figure": "Table 1",
  "page": 6
}
```

Rules:

- `section` is required and non-empty.
- `table_or_figure` may be `null` when the evidence is prose.
- `page` may be `null` only when page-level evidence is not available.
- Anchors should point to the most direct source of the label.
- Invalid citation IDs or missing required anchors are deterministic failures;
  LLM judge scores are secondary.

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

Additional v0.2 fields:

- `readiness_evidence_type`
- `evidence_anchors`
- `evidence_confidence`
- `evidence_notes`

Allowed `maturity_level` values:

- `research_only`
- `experimental`
- `production_ready`

Readiness labels must use only evidence present in the paper or links explicitly
provided by the paper. Do not use later ecosystem adoption, later Hugging Face
ports, or later production usage as ground truth.

## Report Coverage And Judgment

`expected_report_coverage.must_mention` contains central concepts that should
appear in the engineer-facing report. This is a coverage proxy, not a semantic
correctness metric.

`expected_report_judgment` is fixed for the current deterministic dataset
contract:

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
reserved for non-gating LLM judge scoring.

## QA Cases

Each record has exactly three QA cases.

Required v0.1 fields:

- `id`
- `question`
- `expected_answer_keywords`
- `required_citation_paper_ids`
- `min_citations`

Additional required v0.2 fields:

- `question_type`
- `evidence_anchors`
- `must_not_claim`

Optional v0.2 QA review metadata:

- `anchor_source`
- `evidence_confidence`
- `must_not_claim_confidence`
- `must_not_claim_source`
- `review_note`

Observed `question_type` values include:

- `main_contribution`
- `mechanism`
- `limitation`
- `benchmark`
- `comparison`
- `implementation`

For single-paper QA, `required_citation_paper_ids` should be `[paper_id]` and
`min_citations` should be at least `1`.

## Validation Requirements

Dataset validation must produce a structured summary with at least:

- record count
- duplicate paper IDs
- benchmark row count
- QA case count
- dataset version breakdown
- schema version breakdown
- label quality breakdown
- paper family breakdown
- difficulty tag breakdown

Current expected v0.2 summary for `paperintel_60_v2_2.jsonl`:

```text
records=60
duplicates=0
benchmark_rows=333
qa_cases=180
dataset_versions={"v0.2": 60}
schema_versions={"0.2": 60}
label_quality={"manual_verified": 30, "draft_machine": 30}
```

Current v0.1 validation command:

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

Expected output starts with:

```text
OK records=30
```
