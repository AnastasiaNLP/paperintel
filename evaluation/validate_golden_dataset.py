from __future__ import annotations

import argparse

from evaluation.golden_dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenDatasetError,
    summarize_golden_validation,
    validate_golden_file,
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
        validation = validate_golden_file(args.path)
    except GoldenDatasetError as exc:
        print(f"ERROR {exc}")
        return 1

    print(summarize_golden_validation(validation))
    if validation.summary.duplicates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
