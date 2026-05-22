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

The current Evaluation MVP is complete for artifact-level deterministic checks
and manual judge scoring:

- golden seed dataset: ready, with 5 manually verified papers
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

## Inputs

Evaluation uses two JSONL files:

- `golden_dataset/seed_5.jsonl`: manually verified golden labels.
- `workspaces.jsonl`: exported `PaperWorkspace` rows from Postgres.

The golden dataset is local seed data for CI and development. The larger target
dataset is intended to live on Hugging Face.

## Validate Golden Labels

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/seed_5.jsonl
```

Expected output:

```text
OK records=5 paper_ids=1706.03762,2005.11401,2106.09685,2210.03629,2205.14135
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

- The local golden dataset has only 5 seed papers. It is enough for loader,
  runner, and CLI contract tests, but not enough for project-level quality
  claims.
- The target 30-paper dataset is not in the repository yet. It is expected to
  live on Hugging Face, with the local seed kept for CI/development.
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

1. Expand the dataset from 5 local seed papers to a 30-paper Hugging Face
   dataset.
2. Add QA judge task generation for `qa_faithfulness`.
3. Add a judge report artifact format for trend tracking across runs.
4. Add optional scheduled judge evaluation once score variance is understood.
5. Add a deterministic CI job after the repository CI shape is finalized.
