import pytest

from api.in_memory_session_store import InMemorySessionStore
from models.discovery import SearchCandidate
from services.selected_candidate_resolver import (
    NoSelectedCandidatesError,
    SelectedCandidateMissingError,
    SelectedCandidateMissingUrlError,
    SelectedCandidateNotReadyError,
    SelectedCandidateResolver,
)


class FakeCandidateRepository:
    def __init__(self, candidates: list[SearchCandidate]) -> None:
        self.candidates = {candidate.id: candidate for candidate in candidates}
        self.requests = []

    def get_many_by_ids(self, candidate_ids: list[str]) -> list[SearchCandidate]:
        self.requests.append(list(candidate_ids))
        return [
            self.candidates[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in self.candidates
        ]


def _candidate(
    candidate_id: str,
    *,
    status: str = "selected",
    url: str | None = None,
) -> SearchCandidate:
    return SearchCandidate(
        id=candidate_id,
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        status=status,  # type: ignore[arg-type]
        title=f"Paper {candidate_id}",
        url=url if url is not None else f"https://arxiv.org/abs/{candidate_id}",
        arxiv_id=candidate_id,
    )


def _resolver(
    *,
    selected_ids: list[str],
    candidates: list[SearchCandidate],
) -> tuple[SelectedCandidateResolver, str, FakeCandidateRepository]:
    store = InMemorySessionStore()
    session = store.create_session()
    store.set_selected_candidate_ids(session.id, selected_ids)
    repository = FakeCandidateRepository(candidates)
    return (
        SelectedCandidateResolver(
            session_store=store,
            candidate_repository=repository,
        ),
        session.id,
        repository,
    )


def test_resolver_returns_selected_candidates_in_session_order():
    resolver, session_id, repository = _resolver(
        selected_ids=["candidate-2", "candidate-1"],
        candidates=[_candidate("candidate-1"), _candidate("candidate-2")],
    )

    result = resolver.resolve(session_id)

    assert result.session_id == session_id
    assert result.candidate_ids == ["candidate-2", "candidate-1"]
    assert result.urls == [
        "https://arxiv.org/abs/candidate-2",
        "https://arxiv.org/abs/candidate-1",
    ]
    assert repository.requests == [["candidate-2", "candidate-1"]]


def test_resolver_raises_when_no_candidates_selected():
    resolver, session_id, _ = _resolver(selected_ids=[], candidates=[])

    with pytest.raises(NoSelectedCandidatesError):
        resolver.resolve(session_id)


def test_resolver_raises_when_selected_candidate_record_missing():
    resolver, session_id, _ = _resolver(
        selected_ids=["candidate-1", "missing"],
        candidates=[_candidate("candidate-1")],
    )

    with pytest.raises(SelectedCandidateMissingError) as excinfo:
        resolver.resolve(session_id)

    assert excinfo.value.candidate_ids == ["missing"]


def test_resolver_raises_when_candidate_is_not_selected():
    resolver, session_id, _ = _resolver(
        selected_ids=["candidate-1"],
        candidates=[_candidate("candidate-1", status="proposed")],
    )

    with pytest.raises(SelectedCandidateNotReadyError) as excinfo:
        resolver.resolve(session_id)

    assert excinfo.value.candidate.id == "candidate-1"


def test_resolver_raises_for_already_analyzed_candidate():
    resolver, session_id, _ = _resolver(
        selected_ids=["candidate-1"],
        candidates=[_candidate("candidate-1", status="analyzed")],
    )

    with pytest.raises(SelectedCandidateNotReadyError):
        resolver.resolve(session_id)


def test_resolver_raises_when_candidate_url_missing():
    candidate = SearchCandidate.model_construct(
        id="candidate-1",
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        status="selected",
        title="Paper candidate-1",
        url=" ",
        arxiv_id="candidate-1",
    )
    resolver, session_id, _ = _resolver(
        selected_ids=["candidate-1"],
        candidates=[candidate],
    )

    with pytest.raises(SelectedCandidateMissingUrlError):
        resolver.resolve(session_id)
