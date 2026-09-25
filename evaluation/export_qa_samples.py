from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation.golden_dataset import GoldenDatasetError, load_golden_records
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresSessionStore


class QASampleExportError(ValueError):
    """Raised when QA results cannot be matched to golden QA cases."""


def export_qa_samples(
    *,
    store: PostgresSessionStore,
    session_id: str,
    golden_path: str | Path,
    output_path: str | Path,
) -> int:
    records = load_golden_records(golden_path)
    qa_cases = [
        (record.paper_id, case)
        for record in records
        for case in record.qa_cases
    ]
    matching: dict[tuple[str, str], dict] = {}
    for turn in store.list_recent_turns(session_id, limit=100_000):
        if turn.role != "assistant":
            continue
        sample = turn.metadata.get("qa_evaluation_sample")
        if not isinstance(sample, dict):
            continue
        question_key = _normalize_question(str(sample.get("question", "")))
        referenced_ids = set(turn.referenced_paper_ids)
        referenced_ids.update(
            str(citation.get("paper_id"))
            for citation in sample.get("citations", [])
            if isinstance(citation, dict) and citation.get("paper_id")
        )
        candidates = [
            (paper_id, case)
            for paper_id, case in qa_cases
            if _normalize_question(case.question) == question_key
            and (not referenced_ids or paper_id in referenced_ids)
        ]
        if len(candidates) > 1:
            raise QASampleExportError(
                f"Golden question matched multiple papers for turn {turn.id}"
            )
        if not candidates:
            continue
        paper_id, case = candidates[0]
        key = (paper_id, case.id)
        matching[key] = {
            "paper_id": paper_id,
            "qa_case_id": case.id,
            "question": sample.get("question"),
            "answer_text": sample.get("answer_text", ""),
            "citations": sample.get("citations", []),
            "evidence_chunks": sample.get("evidence_chunks", []),
            "intent": sample.get("intent"),
            "persona": sample.get("persona"),
            "repair_iteration": sample.get("repair_iteration", 0),
            "turn_id": turn.id,
        }
    if not matching:
        raise QASampleExportError(
            "No assistant QA turns matched the supplied golden questions. "
            "Questions must match exactly after whitespace and case normalization."
        )
    output = Path(output_path)
    output.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for _, value in sorted(matching.items())
        ),
        encoding="utf-8",
    )
    return len(matching)


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export persisted QA answers matching a golden dataset."
    )
    parser.add_argument("--database-url", required=True, help="Postgres database URL.")
    parser.add_argument("--session-id", required=True, help="Session to export.")
    parser.add_argument("--golden", required=True, help="Golden dataset JSONL path.")
    parser.add_argument("--output", required=True, help="Output QA samples JSONL path.")
    args = parser.parse_args()

    engine = make_engine(args.database_url)
    try:
        count = export_qa_samples(
            store=PostgresSessionStore(make_session_factory(engine)),
            session_id=args.session_id,
            golden_path=args.golden,
            output_path=args.output,
        )
    except (GoldenDatasetError, QASampleExportError) as exc:
        print(f"ERROR {exc}")
        return 1
    finally:
        engine.dispose()
    print(f"Exported {count} QA sample(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
