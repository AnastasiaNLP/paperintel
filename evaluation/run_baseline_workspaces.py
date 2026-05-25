from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.export_workspaces import (
    WorkspaceExportError,
    export_workspaces_for_session,
)
from evaluation.golden_dataset import GoldenDatasetError, load_golden_records


DEFAULT_GOLDEN_PATH = Path("golden_dataset/paperintel_30_v0_1.jsonl")
DEFAULT_OUTPUT_PATH = Path("/tmp/paperintel_30_workspaces.jsonl")


@dataclass(frozen=True)
class BaselinePaper:
    paper_id: str
    source_url: str
    title: str
    metadata_fallback: dict[str, object]


@dataclass(frozen=True)
class BaselineRunResult:
    session_id: str
    requested_count: int
    analyzed_count: int
    failed_count: int
    exported_count: int
    output_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, analyze, and export PaperIntel workspaces for a golden "
            "baseline dataset."
        )
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_PATH),
        help="Golden dataset JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Exported workspace JSONL output path.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override Postgres URL. Defaults to configured PAPERINTEL settings.",
    )
    parser.add_argument(
        "--upgrade-db",
        action="store_true",
        help="Run Alembic upgrade head before creating the PaperIntel service.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Override Qdrant URL. Defaults to configured PAPERINTEL settings.",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Override Qdrant collection. Defaults to configured PAPERINTEL settings.",
    )
    parser.add_argument(
        "--persona",
        default="engineer",
        choices=["engineer", "researcher", "founder"],
        help="Session persona.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N golden records.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Analyze only selected paper_id values. Repeat for multiple papers.",
    )
    parser.add_argument(
        "--resume-session-id",
        default=None,
        help=(
            "Reuse an existing session and export from it after analysis. "
            "If omitted, a new session is created."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have a workspace in the target session.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Delay between paper analyses to reduce external rate-limit pressure.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue analyzing later papers after one paper fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned papers without DB, Qdrant, or LLM calls.",
    )
    args = parser.parse_args()

    try:
        papers = load_baseline_papers(
            args.golden,
            selected_paper_ids=args.paper_id,
            limit=args.limit,
        )
    except GoldenDatasetError as exc:
        print(f"ERROR {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 1

    print(f"BASELINE_GOLDEN={args.golden}")
    print(f"BASELINE_PAPER_COUNT={len(papers)}")
    for index, paper in enumerate(papers, start=1):
        print(f"BASELINE_PAPER_{index}={paper.paper_id} {paper.source_url}")

    if args.dry_run:
        print("BASELINE_DRY_RUN=1")
        return 0

    if args.upgrade_db:
        upgrade_database(args.database_url)

    from api.app_factory import create_paperintel_service

    service = create_paperintel_service(
        database_url=args.database_url,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    try:
        result = run_baseline(
            service=service,
            papers=papers,
            output_path=Path(args.output),
            persona=args.persona,
            resume_session_id=args.resume_session_id,
            skip_existing=args.skip_existing,
            sleep_seconds=args.sleep_seconds,
            continue_on_error=args.continue_on_error,
        )
    except WorkspaceExportError as exc:
        print(f"ERROR export failed: {exc}")
        return 1

    print(f"BASELINE_SESSION_ID={result.session_id}")
    print(f"BASELINE_ANALYZED_COUNT={result.analyzed_count}")
    print(f"BASELINE_FAILED_COUNT={result.failed_count}")
    print(f"BASELINE_EXPORTED_COUNT={result.exported_count}")
    print(f"BASELINE_OUTPUT={result.output_path}")
    return 0 if result.failed_count == 0 else 2


def load_baseline_papers(
    golden_path: str | Path,
    *,
    selected_paper_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[BaselinePaper]:
    records = load_golden_records(golden_path)
    selected = list(dict.fromkeys(selected_paper_ids or []))
    if selected:
        by_id = {record.paper_id: record for record in records}
        missing = [paper_id for paper_id in selected if paper_id not in by_id]
        if missing:
            raise ValueError(
                "Requested paper_id values are not in the golden dataset: "
                + ",".join(missing)
            )
        records = [by_id[paper_id] for paper_id in selected]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        records = records[:limit]
    return [
        BaselinePaper(
            paper_id=record.paper_id,
            source_url=record.source_url,
            title=record.title,
            metadata_fallback={
                "title": record.title,
                "authors": [],
                "arxiv_id": record.paper_id,
                "published_date": "",
                "abstract": "",
                "categories": [],
            },
        )
        for record in records
    ]


def run_baseline(
    *,
    service: Any,
    papers: list[BaselinePaper],
    output_path: Path,
    persona: str,
    resume_session_id: str | None,
    skip_existing: bool,
    sleep_seconds: float,
    continue_on_error: bool,
) -> BaselineRunResult:
    if service.artifact_repository is None:
        raise RuntimeError("Paper workspace repository is not configured.")
    service.handler.analysis_metadata_fallback_by_arxiv_id = {
        paper.paper_id: paper.metadata_fallback for paper in papers
    }

    session_id = resume_session_id
    if session_id is None:
        session = service.create_session(
            persona=persona,
            original_query="PaperIntel 30-paper deterministic baseline",
        )
        session_id = session.id
        print(f"BASELINE_CREATED_SESSION_ID={session_id}", flush=True)
    else:
        service.get_session(session_id)
        print(f"BASELINE_RESUME_SESSION_ID={session_id}", flush=True)

    existing_ids = set()
    if skip_existing:
        existing_ids = {workspace.paper_id for workspace in service.list_paper_workspaces(session_id)}
        print(f"BASELINE_EXISTING_WORKSPACES={','.join(sorted(existing_ids))}", flush=True)

    analyzed = 0
    failed: list[dict[str, str]] = []
    for index, paper in enumerate(papers, start=1):
        if paper.paper_id in existing_ids:
            print(
                f"BASELINE_SKIP_EXISTING index={index} paper_id={paper.paper_id}",
                flush=True,
            )
            continue

        started = time.monotonic()
        print(
            f"BASELINE_ANALYZE_START index={index}/{len(papers)} "
            f"paper_id={paper.paper_id} url={paper.source_url}",
            flush=True,
        )
        try:
            result = service.analyze_paper(session_id, paper.source_url)
            elapsed = time.monotonic() - started
            print(
                f"BASELINE_ANALYZE_DONE paper_id={paper.paper_id} "
                f"phase={result.phase} elapsed={elapsed:.1f}s "
                f"errors={len(result.errors)}",
                flush=True,
            )
            if result.phase == "failed" or result.error is not None:
                failed.append(
                    {
                        "paper_id": paper.paper_id,
                        "reason": result.error.message if result.error else result.phase,
                    }
                )
                if not continue_on_error:
                    break
            else:
                analyzed += 1
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(
                f"BASELINE_ANALYZE_ERROR paper_id={paper.paper_id} "
                f"elapsed={elapsed:.1f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            failed.append({"paper_id": paper.paper_id, "reason": str(exc)})
            if not continue_on_error:
                break

        if sleep_seconds > 0 and index < len(papers):
            time.sleep(sleep_seconds)

    requested_ids = [paper.paper_id for paper in papers]
    exported = export_workspaces_for_session(
        repository=service.artifact_repository,
        session_id=session_id,
        output_path=output_path,
        paper_ids=requested_ids if not failed else None,
    )
    _write_run_summary(
        output_path=output_path,
        session_id=session_id,
        requested=papers,
        exported_count=len(exported),
        failed=failed,
    )
    return BaselineRunResult(
        session_id=session_id,
        requested_count=len(papers),
        analyzed_count=analyzed,
        failed_count=len(failed),
        exported_count=len(exported),
        output_path=output_path,
    )


def upgrade_database(database_url: str | None = None) -> None:
    from alembic import command
    from alembic.config import Config

    resolved_database_url = database_url
    if resolved_database_url is None:
        from config.settings import settings

        resolved_database_url = settings.postgres_url

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", resolved_database_url)
    command.upgrade(config, "head")
    print("BASELINE_DB_UPGRADE=head", flush=True)


def _write_run_summary(
    *,
    output_path: Path,
    session_id: str,
    requested: list[BaselinePaper],
    exported_count: int,
    failed: list[dict[str, str]],
) -> None:
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary = {
        "session_id": session_id,
        "requested_count": len(requested),
        "requested_paper_ids": [paper.paper_id for paper in requested],
        "exported_count": exported_count,
        "failed_count": len(failed),
        "failed": failed,
        "workspace_output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"BASELINE_SUMMARY={summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
