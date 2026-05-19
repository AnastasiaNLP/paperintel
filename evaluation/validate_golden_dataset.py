from __future__ import annotations

import argparse

from evaluation.golden_dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenDatasetError,
    load_golden_records,
    summarize_golden_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PaperIntel golden dataset JSONL.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_GOLDEN_DATASET_PATH),
        help="Path to a golden dataset JSONL file.",
    )
    args = parser.parse_args()

    try:
        records = load_golden_records(args.path)
    except GoldenDatasetError as exc:
        print(f"ERROR {exc}")
        return 1

    print(summarize_golden_records(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

