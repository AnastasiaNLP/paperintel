# Hugging Face Publish Checklist

This checklist prepares the 30-paper golden dataset for a Hugging Face Dataset
repository.

Published dataset:

```text
https://huggingface.co/datasets/AIAnastasia/arxiv-papers
```

## Files To Upload

Upload these files to the Hugging Face dataset repo:

- `paperintel_30_v0_1.jsonl`
- `README.md` copied from `HF_DATASET_CARD.md`

Optional supporting files:

- `SCHEMA.md`
- `seed_5.jsonl`

The source repository keeps the card as `HF_DATASET_CARD.md` to avoid confusing
it with the local `golden_dataset/README.md`. In the Hugging Face dataset repo,
the card should be named `README.md`.

## Pre-Publish Validation

From the PaperIntel repository:

```bash
.venv/bin/python -m evaluation.validate_golden_dataset golden_dataset/paperintel_30_v0_1.jsonl
```

Expected:

```text
OK records=30
```

## Suggested Dataset Repository Name

Hugging Face dataset repo:

```text
paperintel-30-golden-eval
```

## License Decision

`HF_DATASET_CARD.md` currently uses:

```yaml
license: other
```

This is conservative because source papers retain their original licenses and
the dataset only contains manually created labels plus paper metadata. Before
publication, decide whether the annotation layer should use a more specific
license, for example `cc-by-4.0`.

## Scripted Upload

The repository includes a small upload helper. First validate without network
access:

```bash
.venv/bin/python scripts/upload_golden_dataset_to_hf.py --dry-run
```

Then upload to the existing dataset repo:

```bash
.venv/bin/python -m pip install huggingface_hub
HF_TOKEN=... .venv/bin/python scripts/upload_golden_dataset_to_hf.py
```

The default repo id is:

```text
AIAnastasia/arxiv-papers
```

To upload the local seed file too:

```bash
HF_TOKEN=... .venv/bin/python scripts/upload_golden_dataset_to_hf.py --include-seed
```

The script uploads:

- `paperintel_30_v0_1.jsonl`
- `README.md` copied from `HF_DATASET_CARD.md`
- `SCHEMA.md`

## Manual Publish Steps

1. Create a new Hugging Face dataset repository.
2. Copy `golden_dataset/HF_DATASET_CARD.md` to the dataset repo as `README.md`.
3. Copy `golden_dataset/paperintel_30_v0_1.jsonl` to the dataset repo root.
4. Optionally copy `golden_dataset/SCHEMA.md`.
5. Run the Hugging Face preview/load check in the web UI.
6. Confirm the public URL is documented in the PaperIntel README files.

## Post-Publish Repository Update

After publishing, keep these files aligned with the public dataset URL:

- `README.md`
- `evaluation/README.md`
- `golden_dataset/README.md`

Keep `seed_5.jsonl` documented as the local fast test subset.
