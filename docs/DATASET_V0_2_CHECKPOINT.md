# Dataset v0.2 Checkpoint

This checkpoint records the current PaperIntel dataset state before starting
`paperintel_eval_v1`.

## Current Status

- Dataset status: COMPLETE (v0.2)
- Next workstream: Evaluation Framework (`paperintel_eval_v1`)
- Main dataset status: `review_ready`

Main dataset:

```text
golden_dataset/paperintel_60_v2_2.jsonl
```

Summary:

- 60 papers
- 333 benchmark rows
- 180 QA cases
- 0 placeholder issues
- 0 duplicates

## Sources

### First 30 Papers

File:

```text
paperintel_30_v0_2_manual_reviewed_clean.jsonl
```

Status: `manual_verified`

Coverage:

- 30 papers
- 217 benchmark rows
- 90 QA cases
- 100% anchor validation
- manual spot-check completed

Manually checked examples:

- Attention Is All You Need
- LoRA
- QLoRA
- FlashAttention
- WebArena
- SELF-RAG
- BERT

### Second 30 Papers

File:

```text
paperintel_30_new_v0_2_review_ready.jsonl
```

Status: `review_ready`

Coverage:

- 30 papers
- 116 benchmark rows
- 90 QA cases
- schema-valid
- readiness audit completed

### Merged Dataset

File:

```text
golden_dataset/paperintel_60_v2_2.jsonl
```

Validation:

- records: 60
- duplicates: 0
- dataset_version: `v0.2`
- schema_version: `0.2`

Note: the current repository validator is still strict for the v0.1 schema and
does not yet accept v0.2 enrichment fields such as evidence anchors, QA
question types, difficulty tags, and review metadata. Updating the golden
dataset schema/loader is part of the next evaluation framework workstream.

## Coverage

Paper families:

```text
architecture    13
agents          12
evals            7
retrieval        5
serving          5
memory           5
multimodal       5
training         4
alignment        3
rag              1
```

Benchmarks:

```text
total benchmark rows: 333
mean per paper: 5.55
min: 3
max: 8
```

QA:

```text
180 QA cases total
3 QA per paper
```

QA types:

```text
main_contribution : 60
mechanism         : 58
limitation        : 38
benchmark         : 18
comparison        : 5
implementation    : 1
```

Readiness:

```text
paper_explicit            21
paper_linked_repository   12
external_from_v01         11
not_found                 12
external_from_seed         4
```

Difficulty tags:

```text
table_heavy      52
multi_dataset    39
long_appendix     6
no_code           5
ambiguous_units   4
systems_metrics   4
formula_heavy     2
```

## Data Quality

Placeholder issues: 0

Checked categories:

- unsupported claim placeholders: OK
- paper section name placeholders: OK
- TODO placeholders: OK
- schema validation: OK
- anchors validation: OK

## Next Workstream

Target:

```text
paperintel_eval_v1
```

Planned package:

```text
eval/
├── method_eval.py
├── benchmark_eval.py
├── readiness_eval.py
├── report_eval.py
├── hallucination_eval.py
└── run_eval.py
```

Initial MVP:

1. `method_eval`
2. `benchmark_eval`

First-run test set:

- Attention Is All You Need
- LoRA
- QLoRA
- FlashAttention
- SELF-RAG
