# PaperIntel Evaluation

This package contains the local deterministic evaluation path for persisted
PaperIntel artifacts. It intentionally avoids live model calls and DeepEval for
now; LLM-judge metrics will be layered on top of this once the deterministic
artifact checks are stable.

The deterministic runner is a CI-suitable gate for stable, repeatable checks.
Future LLM-judge evaluation is a gauge: it reports quality signals and trends,
but should not be part of normal CI pass/fail because judge scores are
non-deterministic.

## Evaluation MVP Status

The Evaluation MVP is closed for dataset, artifact-level deterministic checks,
workspace export, and manual judge scoring:

- golden seed dataset: ready, with 5 manually verified papers
- 30-paper golden dataset: ready, schema-clean, and published on Hugging Face
- golden loader and schema validation: ready
- deterministic artifact metrics: ready
- workspace export from Postgres: ready
- file-based deterministic runner: ready
- reproducible workspace fixture and CLI contract tests: ready
- rubric files: ready and versioned
- judge dry-run task generation: ready
- live judge provider: ready for explicit manual use

Normal CI should use deterministic validation and deterministic runner tests.
Live judge scoring should remain manual or scheduled until score variance is
measured on the larger dataset.

## Eval Stage Closeout

Closed in this stage:

- `seed_5.jsonl` remains the fast local seed for loader, runner, and fixture
  tests.
- `paperintel_30_v0_1.jsonl` is the schema-clean 30-paper golden dataset and is
  published as
  [AIAnastasia/arxiv-papers](https://huggingface.co/datasets/AIAnastasia/arxiv-papers).
- `golden_dataset/SCHEMA.md` documents the v0.1 dataset contract.
- `golden_dataset/HF_DATASET_CARD.md` and `golden_dataset/HF_PUBLISH.md`
  document the external dataset package and upload flow.
- Deterministic evaluation can validate exported `PaperWorkspace` JSONL files
  without live model calls.
- Judge rubrics are versioned repository artifacts, and judge scoring is
  available as an explicit manual gauge.

Still deferred:

- QA faithfulness judge task generation is not wired yet.
- Judge scores are not CI gates.
- Benchmark extraction quality remains a measured weakness, especially on
  complex PDF tables and systems/alignment papers.

The 30-paper baseline has been run and exported successfully. The baseline is
not presented as high model quality; it is presented as a measured v0.1 quality
profile with explicit weaknesses and next targets.

## Inputs

Evaluation uses two JSONL files:

- `golden_dataset/seed_5.jsonl`: 5-paper manually verified local seed for fast
  development and CI-style contract checks.
- `golden_dataset/paperintel_30_v0_1.jsonl`: 30-paper manually verified dataset
  for portfolio/Hugging Face publication and project-level evaluation. Published
  dataset:
  [AIAnastasia/arxiv-papers](https://huggingface.co/datasets/AIAnastasia/arxiv-papers).
- `workspaces.jsonl`: exported `PaperWorkspace` rows from Postgres.

The local seed keeps CI and development independent from network access. The
30-paper dataset is small enough to keep versioned in the repository and is also
published to Hugging Face as the external portfolio dataset.

## Validate Golden Labels

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/seed_5.jsonl
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

Expected output:

```text
OK records=5 paper_ids=1706.03762,2005.11401,2106.09685,2210.03629,2205.14135
OK records=30 ...
```

## Try Without Postgres

The repository includes a small workspace fixture for CLI contract checks:

```bash
.venv/bin/python -m evaluation.run_deterministic_eval \
  --golden golden_dataset/seed_5.jsonl \
  --workspaces tests/fixtures/evaluation/workspaces_seed_sample.jsonl
```

This fixture intentionally covers only two of the five golden records, and one
workspace is partial. The command should run successfully but return exit code
`2`, indicating that deterministic evaluation completed with missing or failed
checks.

## Export Workspaces

After analyzing papers in a session, export persisted artifacts from Postgres:

```bash
.venv/bin/python -m evaluation.export_workspaces \
  --database-url "$PAPERINTEL_DATABASE_URL" \
  --session-id "$SESSION_ID" \
  --output /tmp/paperintel-workspaces.jsonl
```

## Generate 30-Paper Baseline Workspaces

The baseline helper combines three steps in one command:

1. read paper URLs from `paperintel_30_v0_1.jsonl`;
2. analyze those papers into a PaperIntel session;
3. export persisted `PaperWorkspace` rows to JSONL.

Preview the planned papers without DB, Qdrant, or model calls:

```bash
.venv/bin/python evaluation/run_baseline_workspaces.py --dry-run
```

Run a small smoke batch first:

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/python evaluation/run_baseline_workspaces.py \
    --upgrade-db \
    --limit 2 \
    --output /tmp/paperintel_2_workspaces.jsonl \
    --sleep-seconds 10 \
    --continue-on-error
```

Run the full 30-paper baseline:

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/python evaluation/run_baseline_workspaces.py \
    --upgrade-db \
    --output /tmp/paperintel_30_workspaces.jsonl \
    --sleep-seconds 10 \
    --continue-on-error
```

For a reproducible baseline, prefer local PDFs and golden metadata fallbacks:

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/python evaluation/run_baseline_workspaces.py \
    --upgrade-db \
    --pdf-dir ~/Desktop/pdfs \
    --require-local-pdfs \
    --metadata-source golden \
    --output /tmp/paperintel_30_workspaces.jsonl \
    --sleep-seconds 10 \
    --continue-on-error
```

Local PDFs must be named `<paper_id>.pdf`, for example `2103.00020.pdf`.
`--metadata-source golden` uses the golden dataset metadata fallback and skips
arXiv metadata fetches during ingestion. This keeps the eval input stable and
reduces dependence on arXiv API availability. Semantic Scholar enrichment may
still run as best-effort non-blocking enrichment.

To resume a partially completed run:

```bash
.venv/bin/python -m dotenv run -- \
  .venv/bin/python evaluation/run_baseline_workspaces.py \
    --upgrade-db \
    --resume-session-id "$SESSION_ID" \
    --skip-existing \
    --pdf-dir ~/Desktop/pdfs \
    --metadata-source golden \
    --output /tmp/paperintel_30_workspaces.jsonl \
    --sleep-seconds 10 \
    --continue-on-error
```

The command also writes a sibling summary file:

```text
/tmp/paperintel_30_workspaces.jsonl.summary.json
```

When any paper fails and `--continue-on-error` is set, the export contains the
workspaces that were successfully persisted in the session. Re-run with
`--resume-session-id` and `--skip-existing` after external failures.

To export only selected papers:

```bash
.venv/bin/python -m evaluation.export_workspaces \
  --database-url "$PAPERINTEL_DATABASE_URL" \
  --session-id "$SESSION_ID" \
  --paper-id 1706.03762 \
  --paper-id 2005.11401 \
  --output /tmp/paperintel-workspaces.jsonl
```

## Run Deterministic Evaluation

```bash
.venv/bin/python -m evaluation.run_deterministic_eval \
  --golden golden_dataset/seed_5.jsonl \
  --workspaces /tmp/paperintel-workspaces.jsonl
```

Use `--json` for machine-readable output:

```bash
.venv/bin/python -m evaluation.run_deterministic_eval \
  --golden golden_dataset/seed_5.jsonl \
  --workspaces /tmp/paperintel-workspaces.jsonl \
  --json
```

Exit codes:

- `0`: evaluation ran and all matched records passed.
- `1`: input loading or validation failed.
- `2`: evaluation ran, but some records are missing or some checks failed.

## Baseline Results v0.1

The first 30-paper baseline used `golden_dataset/paperintel_30_v0_1.jsonl` and
an exported `PaperWorkspace` JSONL from a live PaperIntel run.

Workspace coverage:

- matched workspaces: `30/30`
- missing workspaces: `0`

Deterministic evaluation after benchmark matcher normalization:

- average score: `0.2659`
- benchmark score: `0.1127`
- matched workspaces: `30/30`
- missing workspaces: `0`

A targeted local-PDF rerun on the 12 papers that previously had empty benchmark
extraction produced:

- analyzed: `12`
- failed: `0`
- exported: `12`
- empty benchmark papers: `11 -> 9`
- non-empty benchmark papers: `1 -> 3`
- subset benchmark score: `0.0 -> 0.0417`

Manual benchmark extraction review over the 30-paper seed:

- empty benchmark extraction: `11/30`
- clean benchmark extraction: `11/30`
- partial benchmark extraction: `6/30`
- corrupt/noisy benchmark extraction: `2/30`

Interpretation: benchmark extraction is the weakest measured component in
v0.1. This is a useful result of the evaluation stage, not a hidden caveat. The
system now has a reproducible way to expose and track this weakness.

## Deterministic Checks

Current deterministic checks cover:

- method extraction: method name, description keywords, novelty keywords,
  components, baselines, stated limitations
- benchmarks: task, metric, value, unit, and condition keyword coverage
- readiness: open code, code/model links, framework integrations, dependencies,
  GPU requirement, maturity level
- report coverage: required engineer-report concepts

Interpret these checks narrowly:

- Benchmark matching is a strong deterministic metric for discrete reported
  facts.
- Readiness matching is a strong structural metric for explicit artifact fields.
- Method extraction and report keyword checks are coverage proxies only. They
  verify that expected concepts appear in free text, but they do not prove that
  the surrounding statement is semantically correct.

For example, a report could mention `self-attention` and still make an incorrect
claim about it. Deterministic keyword recall would see coverage, not semantic
correctness. Free-text correctness belongs to the LLM-judge layer.

Subjective report fields such as `recommended_action`,
`implementation_difficulty`, and `action_reasoning` are not scored here. They
are reserved for later G-Eval/DeepEval rubric checks.

## Judge Rubrics

Rubrics for the future G-Eval/DeepEval layer live in
`evaluation/rubrics/`. These files are versioned source-of-truth artifacts, not
just documentation. Changing a rubric changes judge behavior and must be treated
like changing a prompt or schema.

The planned judge layer should:

- read rubric files from the repository;
- report scores as observations/trends;
- run manually or on a scheduled workflow, not in normal CI;
- avoid external knowledge unless the rubric explicitly permits it;
- keep deterministic eval and judge eval as separate runners with separate
  pass/fail semantics.

Build judge tasks without making model calls:

```bash
.venv/bin/python -m evaluation.run_judge_eval \
  --golden golden_dataset/seed_5.jsonl \
  --workspaces tests/fixtures/evaluation/workspaces_seed_sample.jsonl \
  --dry-run
```

The dry-run output is JSON. It includes rubric IDs, rubric hashes, paper IDs,
input refs, and `not_scored` results. This verifies judge data plumbing while
keeping the evaluation deterministic.

Run live judge scoring explicitly:

```bash
.venv/bin/python -m evaluation.run_judge_eval \
  --golden golden_dataset/seed_5.jsonl \
  --workspaces tests/fixtures/evaluation/workspaces_seed_sample.jsonl \
  --live
```

Live judge mode calls the configured LLM provider and returns JSON results with
`scored` or `error` statuses. It still exits `0` when scoring completes, even if
scores are low, because judge evaluation is a gauge rather than a CI gate. Input
loading failures and provider setup failures still return exit code `1`.

## Known Limitations

- The 5-paper seed is enough for loader, runner, and CLI contract tests, but not
  enough for project-level quality claims.
- The 30-paper dataset is manually verified and schema-clean, but still small
  and curated. Treat it as a focused evaluation corpus, not a broad benchmark of
  general LLM paper understanding.
- Benchmark extraction is weak in v0.1. Observed failure modes include:
  empty extraction on papers with clear benchmark tables, partial extraction
  that misses headline rows or variants, and noisy extraction of auxiliary or
  ambiguous values.
- The deterministic benchmark matcher now normalizes common aliases but still
  requires task, metric, and value agreement. Low benchmark scores should be
  read as extraction quality gaps, not just matcher artifacts.
- Method and report keyword checks are coverage proxies, not semantic
  correctness checks.
- Judge scores are non-deterministic and should not be used as normal CI gates.
- Live judge scoring currently covers report rubrics:
  `recommended_action`, `implementation_difficulty`, and `action_reasoning`.
- The `qa_faithfulness` rubric exists, but QA judge task generation is not wired
  yet.
- The deterministic runner evaluates exported `PaperWorkspace` JSONL files. It
  does not run paper analysis itself.

## Next Steps

1. Improve benchmark extraction on complex PDF tables and systems/alignment
   papers.
2. Reduce partial/corrupt benchmark rows by strengthening unit, condition, and
   headline-row handling.
3. Add QA judge task generation for `qa_faithfulness`.
4. Add citation-grounding metrics.
5. Add comparison/synthesis evaluation once the comparison analyst and synthesis
   agent land.
6. Revisit benchmark extraction after expanding beyond the 30-paper seed
   dataset, to avoid overfitting prompt changes to the current corpus.
