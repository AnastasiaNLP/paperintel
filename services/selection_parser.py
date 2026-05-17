import re
from dataclasses import dataclass, field
from typing import Protocol

from api.session_store import SessionStore
from models.discovery import SearchCandidate, SelectionSet


class SearchCandidateRepository(Protocol):
    def list_latest_for_session(self, session_id: str) -> list[SearchCandidate]:
        ...

    def update_status(
        self,
        candidate_id: str,
        status: str,
    ) -> SearchCandidate | None:
        ...


@dataclass
class SelectionParseResult:
    display_ranks: list[int]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class SelectionHandlingResult:
    selection: SelectionSet | None
    candidates: list[SearchCandidate]
    response_text: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.selection is not None and not self.errors


def parse_selection_ranks(message: str) -> SelectionParseResult:
    ranks = [int(match) for match in re.findall(r"\d+", message)]
    deduped = list(dict.fromkeys(ranks))
    if not deduped:
        return SelectionParseResult(
            display_ranks=[],
            errors=["No selection numbers found. Reply with numbers like 1, 3, 5."],
        )
    if any(rank < 1 for rank in deduped):
        return SelectionParseResult(
            display_ranks=[],
            errors=["Selection numbers must be 1-based."],
        )
    return SelectionParseResult(display_ranks=deduped)


def resolve_selection(
    *,
    session_id: str,
    message: str,
    candidates: list[SearchCandidate],
) -> SelectionHandlingResult:
    parsed = parse_selection_ranks(message)
    if not parsed.ok:
        return SelectionHandlingResult(
            selection=None,
            candidates=[],
            response_text=parsed.errors[0],
            errors=parsed.errors,
        )

    candidates_by_rank = {candidate.display_rank: candidate for candidate in candidates}
    missing = [rank for rank in parsed.display_ranks if rank not in candidates_by_rank]
    if missing:
        available = ", ".join(
            str(candidate.display_rank)
            for candidate in sorted(candidates, key=lambda item: item.display_rank)
        )
        error_text = (
            f"I could not find candidates numbered {', '.join(map(str, missing))}. "
            f"Available numbers are: {available or 'none'}."
        )
        return SelectionHandlingResult(
            selection=None,
            candidates=[],
            response_text=error_text,
            errors=[error_text],
        )

    selected = [candidates_by_rank[rank] for rank in parsed.display_ranks]
    return SelectionHandlingResult(
        selection=SelectionSet(
            session_id=session_id,
            discovery_turn_id=selected[0].discovery_turn_id,
            selected_candidate_ids=[candidate.id for candidate in selected],
            display_ranks=parsed.display_ranks,
        ),
        candidates=selected,
        response_text=format_selected_candidates(selected),
    )


def format_selected_candidates(candidates: list[SearchCandidate]) -> str:
    lines = [f"Selected {len(candidates)} paper{'s' if len(candidates) != 1 else ''}:"]
    for candidate in candidates:
        label = f"[{candidate.display_rank}] {candidate.title}"
        if candidate.year:
            label = f"{label} ({candidate.year})"
        lines.append(f"- {label} — {candidate.url}")
    lines.extend(
        [
            "",
            "Send these URLs to analyze them, or ask me to analyze selected papers in the next step.",
        ]
    )
    return "\n".join(lines)


class SelectionHandler:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        candidate_repository: SearchCandidateRepository,
    ) -> None:
        self.session_store = session_store
        self.candidate_repository = candidate_repository

    def handle(self, *, session_id: str, message: str) -> SelectionHandlingResult:
        candidates = self.candidate_repository.list_latest_for_session(session_id)
        result = resolve_selection(
            session_id=session_id,
            message=message,
            candidates=candidates,
        )
        if result.selection is None:
            return result

        for candidate_id in result.selection.selected_candidate_ids:
            self.candidate_repository.update_status(candidate_id, "selected")
        self.session_store.set_selected_candidate_ids(
            session_id,
            result.selection.selected_candidate_ids,
        )
        return result
