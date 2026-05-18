import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from models.discovery import DiscoveryPlan, RawSearchResult, SearchCandidate
from services.search_provider import SearchProvider

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LIMIT = 10
TITLE_MATCH_WEIGHT = 4.0
ABSTRACT_MATCH_WEIGHT = 1.5
PHRASE_MATCH_WEIGHT = 3.0
RECENCY_WEIGHT = 0.05
SOURCE_WEIGHT = 1.0


class SearchCandidateRepository(Protocol):
    def upsert_many(self, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        ...


@dataclass
class SearcherResult:
    candidates: list[SearchCandidate]
    warnings: list[str] = field(default_factory=list)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _normalize_title(value: str | None) -> str:
    normalized = _normalize_text(value)
    return re.sub(r"[^\w\s]", "", normalized)


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = parsed.path.removesuffix(".pdf")
    if path.startswith("/pdf/"):
        path = path.replace("/pdf/", "/abs/", 1)
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path.rstrip("/"),
            "",
            "",
        )
    )


def _dedup_keys(result: RawSearchResult) -> list[tuple[str, str]]:
    keys = []
    if result.arxiv_id:
        keys.append(("arxiv_id", result.arxiv_id.casefold()))
    if result.url:
        keys.append(("url", _canonical_url(result.url)))
    title_key = _normalize_title(result.title)
    if title_key:
        keys.append(("title", title_key))
    return keys


def _query_terms(plan: DiscoveryPlan) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for query in plan.queries:
        for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", query.query.casefold()):
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def _recency_score(year: int | None) -> float:
    if year is None:
        return 0.0
    current_year = datetime.now().year
    age = max(0, current_year - year)
    return max(0.0, 10.0 - age) * RECENCY_WEIGHT


def _year_from_result(result: RawSearchResult) -> int | None:
    if result.year is not None:
        return result.year
    if result.published_at is not None:
        return result.published_at.year
    return None


def _score_result(result: RawSearchResult, plan: DiscoveryPlan) -> tuple[float, list[str]]:
    title = _normalize_text(result.title)
    abstract = _normalize_text(result.abstract)
    reasons: list[str] = []
    score = 0.0

    for query in plan.queries:
        normalized_query = _normalize_text(query.query)
        if normalized_query and normalized_query in title:
            score += PHRASE_MATCH_WEIGHT
            reasons.append("query phrase in title")
            break
        if normalized_query and normalized_query in abstract:
            score += PHRASE_MATCH_WEIGHT / 2
            reasons.append("query phrase in abstract")
            break

    title_matches = 0
    abstract_matches = 0
    for term in _query_terms(plan):
        if term in title:
            title_matches += 1
        elif term in abstract:
            abstract_matches += 1

    if title_matches:
        score += title_matches * TITLE_MATCH_WEIGHT
        reasons.append(f"{title_matches} query terms in title")
    if abstract_matches:
        score += abstract_matches * ABSTRACT_MATCH_WEIGHT
        reasons.append(f"{abstract_matches} query terms in abstract")

    year = _year_from_result(result)
    recency = _recency_score(year)
    if recency:
        score += recency
        reasons.append("recent paper")

    if result.source == "arxiv":
        score += SOURCE_WEIGHT
        reasons.append("arxiv source")

    if not reasons:
        reasons.append("provider result")

    return round(score, 4), reasons


def _candidate_from_result(
    result: RawSearchResult,
    *,
    session_id: str,
    discovery_turn_id: str,
    display_rank: int,
    score: float,
    reasons: list[str],
) -> SearchCandidate:
    return SearchCandidate(
        session_id=session_id,
        discovery_turn_id=discovery_turn_id,
        display_rank=display_rank,
        status="proposed",
        title=result.title,
        url=result.url,
        source=result.source,
        authors=result.authors,
        year=_year_from_result(result),
        arxiv_id=result.arxiv_id,
        abstract=result.abstract,
        published_at=result.published_at,
        score=score,
        reasons=reasons,
        metadata=result.metadata,
    )


class Searcher:
    def __init__(
        self,
        *,
        provider: SearchProvider,
        candidate_repository: SearchCandidateRepository,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.provider = provider
        self.candidate_repository = candidate_repository
        self.candidate_limit = candidate_limit

    def search(
        self,
        *,
        session_id: str,
        discovery_turn_id: str,
        plan: DiscoveryPlan,
    ) -> SearcherResult:
        raw_results: list[RawSearchResult] = []
        warnings: list[str] = []
        for query in plan.queries:
            try:
                raw_results.extend(self.provider.search(query))
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                detail = f"HTTP {status_code}" if status_code else type(exc).__name__
                warning = f"Search query failed ({detail}): {query.query}"
                logger.warning(warning)
                warnings.append(warning)

        deduped: list[RawSearchResult] = []
        seen_keys: set[tuple[str, str]] = set()
        for result in raw_results:
            keys = _dedup_keys(result)
            if any(key in seen_keys for key in keys):
                continue
            deduped.append(result)
            seen_keys.update(keys)

        scored = []
        for result in deduped:
            score, reasons = _score_result(result, plan)
            scored.append((score, result.title.casefold(), result, reasons))

        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates = [
            _candidate_from_result(
                result,
                session_id=session_id,
                discovery_turn_id=discovery_turn_id,
                display_rank=index,
                score=score,
                reasons=reasons,
            )
            for index, (score, _title, result, reasons) in enumerate(
                scored[: self.candidate_limit],
                start=1,
            )
        ]

        persisted = self.candidate_repository.upsert_many(candidates)
        return SearcherResult(candidates=persisted, warnings=warnings)
