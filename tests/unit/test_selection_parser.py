from models.discovery import SearchCandidate
from services.selection_parser import (
    SelectionHandler,
    parse_selection_ranks,
    resolve_selection,
)


class FakeRepository:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.updated = []

    def list_latest_for_session(self, session_id: str):
        return list(self.candidates)

    def update_status(self, candidate_id: str, status: str):
        self.updated.append((candidate_id, status))
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate.model_copy(update={"status": status})
        return None


class FakeSessionStore:
    def __init__(self):
        self.selected_candidate_ids = []
        self.phases = []

    def set_selected_candidate_ids(self, session_id: str, candidate_ids: list[str]):
        self.selected_candidate_ids = list(candidate_ids)
        return None

    def update_phase(self, session_id: str, phase: str):
        self.phases.append(phase)
        return None


def _candidate(rank: int, *, candidate_id: str | None = None) -> SearchCandidate:
    arxiv_id = f"2401.0000{rank}"
    return SearchCandidate(
        id=candidate_id or f"candidate-{rank}",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=rank,
        title=f"Paper {rank}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
        year=2024,
    )


def _candidates(count: int = 5) -> list[SearchCandidate]:
    return [_candidate(rank) for rank in range(1, count + 1)]


def test_parse_selection_ranks_extracts_numbers():
    result = parse_selection_ranks("use 1, 3 and 5")

    assert result.ok is True
    assert result.display_ranks == [1, 3, 5]


def test_parse_selection_ranks_deduplicates_preserving_order():
    result = parse_selection_ranks("1, 3, 1, 2")

    assert result.display_ranks == [1, 3, 2]


def test_parse_selection_ranks_rejects_message_without_numbers():
    result = parse_selection_ranks("use the transformer one")

    assert result.ok is False
    assert "No selection numbers" in result.errors[0]


def test_parse_selection_ranks_rejects_zero_rank():
    result = parse_selection_ranks("use 0")

    assert result.ok is False
    assert "1-based" in result.errors[0]


def test_resolve_selection_maps_ranks_to_candidate_ids():
    result = resolve_selection(
        session_id="session-1",
        message="use 1 and 3",
        candidates=_candidates(),
    )

    assert result.ok is True
    assert result.selection is not None
    assert result.selection.selected_candidate_ids == ["candidate-1", "candidate-3"]
    assert result.selection.display_ranks == [1, 3]
    assert [candidate.id for candidate in result.candidates] == [
        "candidate-1",
        "candidate-3",
    ]


def test_resolve_selection_rejects_unknown_rank():
    result = resolve_selection(
        session_id="session-1",
        message="use 1 and 9",
        candidates=_candidates(),
    )

    assert result.ok is False
    assert result.selection is None
    assert "9" in result.response_text
    assert "Available numbers" in result.response_text


def test_resolve_selection_sorts_available_ranks_in_error_message():
    result = resolve_selection(
        session_id="session-1",
        message="use 9",
        candidates=[_candidate(3), _candidate(1), _candidate(2)],
    )

    assert "Available numbers are: 1, 2, 3." in result.response_text


def test_resolve_selection_response_includes_selected_urls():
    result = resolve_selection(
        session_id="session-1",
        message="use 2",
        candidates=_candidates(),
    )

    assert "[2] Paper 2" in result.response_text
    assert "https://arxiv.org/abs/2401.00002" in result.response_text


def test_selection_handler_updates_statuses_and_session_selection():
    repository = FakeRepository(_candidates())
    store = FakeSessionStore()
    handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )

    result = handler.handle(session_id="session-1", message="use 1, 3")

    assert result.ok is True
    assert repository.updated == [
        ("candidate-1", "selected"),
        ("candidate-3", "selected"),
    ]
    assert store.selected_candidate_ids == ["candidate-1", "candidate-3"]
    assert store.phases == []


def test_selection_handler_does_not_update_on_invalid_selection():
    repository = FakeRepository(_candidates())
    store = FakeSessionStore()
    handler = SelectionHandler(
        session_store=store,
        candidate_repository=repository,
    )

    result = handler.handle(session_id="session-1", message="use 9")

    assert result.ok is False
    assert repository.updated == []
    assert store.selected_candidate_ids == []
    assert store.phases == []
