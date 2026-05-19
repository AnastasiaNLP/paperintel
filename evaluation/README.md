# PaperIntel Evaluation

This package contains the local deterministic evaluation path for persisted
PaperIntel artifacts. It intentionally avoids live model calls and DeepEval for
now; LLM-judge metrics will be layered on top of this once the deterministic
artifact checks are stable.

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

Subjective report fields such as `recommended_action`,
`implementation_difficulty`, and `action_reasoning` are not scored here. They
are reserved for later G-Eval/DeepEval rubric checks.

