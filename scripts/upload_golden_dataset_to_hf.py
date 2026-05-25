from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records


DEFAULT_REPO_ID = "AIAnastasia/arxiv-papers"
DEFAULT_DATASET_PATH = Path("golden_dataset/paperintel_30_v0_1.jsonl")
DEFAULT_CARD_PATH = Path("golden_dataset/HF_DATASET_CARD.md")
DEFAULT_SCHEMA_PATH = Path("golden_dataset/SCHEMA.md")
DEFAULT_SEED_PATH = Path("golden_dataset/seed_5.jsonl")


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    path_in_repo: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload the PaperIntel golden dataset package to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id, e.g. AIAnastasia/arxiv-papers.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the 30-paper golden JSONL dataset.",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=DEFAULT_CARD_PATH,
        help="Path to the local HF dataset card template.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the schema document.",
    )
    parser.add_argument(
        "--include-seed",
        action="store_true",
        help="Also upload seed_5.jsonl as an optional local-development subset.",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="Path to seed_5.jsonl when --include-seed is used.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repo first if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private when used with --create-repo.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN "
            "from the environment/cache."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Upload PaperIntel 30-paper golden dataset",
        help="Commit message for the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print upload plan without calling Hugging Face.",
    )
    args = parser.parse_args()

    try:
        records = load_golden_records(args.dataset)
    except GoldenDatasetError as exc:
        print(f"ERROR validating dataset: {exc}")
        return 1

    upload_items = _build_upload_items(
        dataset_path=args.dataset,
        card_path=args.card,
        schema_path=args.schema,
        include_seed=args.include_seed,
        seed_path=args.seed,
    )
    missing = [str(item.local_path) for item in upload_items if not item.local_path.exists()]
    if missing:
        print("ERROR missing upload file(s):")
        for path in missing:
            print(f"- {path}")
        return 1

    print(f"HF_REPO_ID={args.repo_id}")
    print(f"DATASET_RECORDS={len(records)}")
    print("UPLOAD_PLAN:")
    for item in upload_items:
        print(f"- {item.local_path} -> {item.path_in_repo}")

    if args.dry_run:
        print("DRY_RUN=1")
        return 0

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print(
            "ERROR huggingface_hub is not installed. Install it with "
            "`pip install huggingface_hub` or add it to your environment."
        )
        return 1

    api = HfApi(token=token)
    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    operations = [
        CommitOperationAdd(
            path_in_repo=item.path_in_repo,
            path_or_fileobj=str(item.local_path),
        )
        for item in upload_items
    ]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=args.commit_message,
    )
    for item in upload_items:
        print(f"UPLOADED {item.path_in_repo}")

    print(f"DONE https://huggingface.co/datasets/{args.repo_id}")
    return 0


def _build_upload_items(
    *,
    dataset_path: Path,
    card_path: Path,
    schema_path: Path,
    include_seed: bool,
    seed_path: Path,
) -> list[UploadItem]:
    items = [
        UploadItem(dataset_path, "paperintel_30_v0_1.jsonl"),
        UploadItem(card_path, "README.md"),
        UploadItem(schema_path, "SCHEMA.md"),
    ]
    if include_seed:
        items.append(UploadItem(seed_path, "seed_5.jsonl"))
    return items


if __name__ == "__main__":
    raise SystemExit(main())
