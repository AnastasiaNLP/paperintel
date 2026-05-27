import pytest
from pydantic import ValidationError

from models.external_metadata import ArxivMetadataCacheEntry


def test_arxiv_metadata_cache_entry_defaults_to_diagnostic_only_row():
    entry = ArxivMetadataCacheEntry(arxiv_id="1706.03762")

    assert entry.authors == []
    assert entry.categories == []
    assert entry.fetched_at is None
    assert entry.has_successful_fetch is False
    assert entry.error_count == 0
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None


def test_arxiv_metadata_cache_entry_rejects_negative_error_count():
    with pytest.raises(ValidationError):
        ArxivMetadataCacheEntry(arxiv_id="1706.03762", error_count=-1)
