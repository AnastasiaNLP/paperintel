---
pretty_name: PaperIntel 30-Paper Golden Evaluation Dataset
language:
  - en
license: other
task_categories:
  - question-answering
  - text-generation
  - text-classification
tags:
  - ai-papers
  - llm-evaluation
  - retrieval-augmented-generation
  - agents
  - systems
  - golden-dataset
size_categories:
  - n<1K
---

# PaperIntel 30-Paper Golden Evaluation Dataset

This dataset contains 30 manually verified paper-level golden records for
evaluating PaperIntel, an AI/ML paper analysis system. Each record describes one
research paper and includes expected method extraction labels, benchmark rows,
production-readiness labels, report coverage checks, and grounded QA cases.

The dataset is designed for evaluation of structured paper-analysis artifacts,
not for training a language model.

## Dataset Files

Recommended upload file:

- `paperintel_30_v0_1.jsonl`

Local development seed:

- `seed_5.jsonl`

Each file uses JSONL: one JSON object per line.

## What This Dataset Measures

The deterministic evaluation layer can measure:

- benchmark precision and recall for discrete reported facts;
- readiness field matching for explicit artifact fields;
- method/report keyword recall as a coverage proxy;
- QA keyword and citation coverage.

The dataset deliberately separates deterministic checks from subjective judgment.
Fields such as `recommended_action`, `implementation_difficulty`, and
`action_reasoning` are configured for later LLM-judge / G-Eval scoring and are
not treated as exact deterministic ground truth.

## Limitations

Keyword checks are not semantic correctness checks. A model can mention the right
terms while making an incorrect claim. Free-text correctness should be evaluated
with a separate judge rubric.

Labels are grounded in each paper and links explicitly provided by that paper.
They intentionally avoid retrospective ecosystem knowledge, later production
adoption, and later third-party implementations unless those artifacts are
explicitly present in the source paper.

Source papers retain their original licenses and copyrights. This dataset
contains manually created evaluation labels and paper metadata, not paper text.
Set the final Hugging Face dataset license according to the license you choose
for the annotation layer.

## Schema

Top-level fields:

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

See `SCHEMA.md` in the source repository for the full schema contract and
labeling rules.

## Validation

In the PaperIntel repository:

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

Expected output:

```text
OK records=30
```

## Recommended Use

Use this dataset as:

- a golden label corpus for paper-analysis systems;
- a deterministic evaluation target for benchmark and readiness extraction;
- a seed corpus for manual LLM-judge evaluation of report quality;
- a portfolio artifact demonstrating structured evaluation design for AI/ML
  paper intelligence tools.

Do not use it as a broad benchmark of general LLM paper understanding. The
dataset is small, curated, and focused on systems, retrieval, agents, alignment,
and foundation-model papers.

