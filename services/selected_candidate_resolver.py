from dataclasses import dataclass
from typing import Protocol, Sequence

from api.session_store import SessionStore
from models.discovery import SearchCandidate


class SearchCandidateRepository(Protocol):
    def get_many_by_ids(self, candidate_ids: Sequence[str]) -> list[SearchCandidate]:
        ...


class SelectedCandidateError(ValueError):
    pass


class NoSelectedCandidatesError(SelectedCandidateError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session has no selected papers: {session_id}")
        self.session_id = session_id


class SelectedCandidateMissingError(SelectedCandidateError):
    def __init__(self, candidate_ids: list[str]) -> None:
        joined = ", ".join(candidate_ids)
        super().__init__(f"Selected candidate records are missing: {joined}")
        self.candidate_ids = list(candidate_ids)


class SelectedCandidateNotReadyError(SelectedCandidateError):
    def __init__(self, candidate: SearchCandidate) -> None:
        super().__init__(
            f"Selected candidate is not ready for analysis: {candidate.id} "
            f"(status={candidate.status})"
        )
        self.candidate = candidate


class SelectedCandidateMissingUrlError(SelectedCandidateError):
    def __init__(self, candidate: SearchCandidate) -> None:
        super().__init__(f"Selected candidate has no usable URL: {candidate.id}")
        self.candidate = candidate


@dataclass(frozen=True)
class SelectedCandidateSet:
    session_id: str
    candidates: list[SearchCandidate]

    @property
    def candidate_ids(self) -> list[str]:
        return [candidate.id for candidate in self.candidates]

    @property
    def urls(self) -> list[str]:
        return [candidate.url for candidate in self.candidates]


class SelectedCandidateResolver:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        candidate_repository: SearchCandidateRepository,
    ) -> None:
        self.session_store = session_store
        self.candidate_repository = candidate_repository

    def resolve(self, session_id: str) -> SelectedCandidateSet:
        session = self.session_store.require_session(session_id)
        selected_ids = list(session.selected_candidate_ids)
        if not selected_ids:
            raise NoSelectedCandidatesError(session_id)

        candidates = self.candidate_repository.get_many_by_ids(selected_ids)
        by_id = {candidate.id: candidate for candidate in candidates}
        missing_ids = [
            candidate_id for candidate_id in selected_ids if candidate_id not in by_id
        ]
        if missing_ids:
            raise SelectedCandidateMissingError(missing_ids)

        ordered = [by_id[candidate_id] for candidate_id in selected_ids]
        for candidate in ordered:
            if candidate.status != "selected":
                raise SelectedCandidateNotReadyError(candidate)
            if not candidate.url.strip():
                raise SelectedCandidateMissingUrlError(candidate)

        return SelectedCandidateSet(session_id=session.id, candidates=ordered)
