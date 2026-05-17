from typing import Protocol

from models.discovery import RawSearchResult, ResearchQuery


class SearchProvider(Protocol):
    def search(self, query: ResearchQuery) -> list[RawSearchResult]:
        ...
