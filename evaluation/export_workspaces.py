from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

from models.artifacts import PaperWorkspace
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresPaperWorkspaceRepository


class WorkspaceRepository(Protocol):
    def list_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        ...


class WorkspaceExportError(ValueError):
    """Raised when workspace export inputs are invalid."""


def export_workspaces_for_session(
    *,
    repository: WorkspaceRepository,
    session_id: str,
    output_path: str | Path,
    paper_ids: list[str] | None = None,
) -> list[PaperWorkspace]:
    workspaces = repository.list_workspaces(session_id)
    selected = _filter_workspaces(workspaces, paper_ids or [])
    if not selected:
        raise WorkspaceExportError(
            f"No paper workspaces found for session {session_id}"
            + (f" and paper_ids {paper_ids}" if paper_ids else "")
        )

    path = Path(output_path)
    path.write_text(dump_workspaces_jsonl(selected), encoding="utf-8")
    return selected


def dump_workspaces_jsonl(workspaces: list[PaperWorkspace]) -> str:
    return "".join(
        json.dumps(workspace.model_dump(mode="json"), sort_keys=True) + "\n"
        for workspace in workspaces
    )


def _filter_workspaces(
    workspaces: list[PaperWorkspace],
    paper_ids: list[str],
) -> list[PaperWorkspace]:
    if not paper_ids:
        return workspaces

    requested = list(dict.fromkeys(paper_ids))
    by_paper_id = {workspace.paper_id: workspace for workspace in workspaces}
    missing = [paper_id for paper_id in requested if paper_id not in by_paper_id]
    if missing:
        raise WorkspaceExportError(
            "Requested paper workspaces were not found: " + ",".join(missing)
        )
    return [by_paper_id[paper_id] for paper_id in requested]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export persisted PaperWorkspace rows to JSONL for evaluation."
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres database URL.",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="PaperIntel session ID to export.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Optional paper_id to export. Repeat to export multiple papers.",
    )
    args = parser.parse_args()

    engine = make_engine(args.database_url)
    try:
        repository = PostgresPaperWorkspaceRepository(make_session_factory(engine))
        selected = export_workspaces_for_session(
            repository=repository,
            session_id=args.session_id,
            output_path=args.output,
            paper_ids=args.paper_id,
        )
    except WorkspaceExportError as exc:
        print(f"ERROR {exc}")
        return 1
    finally:
        engine.dispose()

    print(f"Exported {len(selected)} workspace(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

