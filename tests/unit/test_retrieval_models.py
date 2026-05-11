import pytest
from pydantic import ValidationError

from models.retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    ChunkLocation,
    ChunkSearchQuery,
    ChunkSource,
    EvidenceArtifact,
    PaperChunk,
)


def test_paper_chunk_defaults_fix_stage_c_embedding_contract():
    chunk = PaperChunk(
        paper_id="2310.06825",
        chunk_index=0,
        text="A retrieval chunk.",
        source=ChunkSource(paper_id="2310.06825", arxiv_id="2310.06825"),
    )

    assert chunk.chunk_type == "text"
    assert chunk.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert chunk.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert chunk.paper_id == "2310.06825"
    assert chunk.source.paper_id == "2310.06825"


def test_chunk_preserves_location_and_evidence_artifact_metadata():
    artifact = EvidenceArtifact(
        paper_id="2310.06825",
        artifact_type="table",
        storage_ref="s3://paperintel/table-1.png",
        label="Table 1",
        page=4,
        bbox={"x0": 10.0, "y0": 20.0, "x1": 300.0, "y1": 180.0},
    )
    chunk = PaperChunk(
        paper_id="2310.06825",
        chunk_index=2,
        text="Table 1 reports accuracy.",
        chunk_type="table",
        source=ChunkSource(paper_id="2310.06825", title="Example Paper"),
        location=ChunkLocation(page_start=4, page_end=4, section_title="Results"),
        artifact_refs=[artifact],
        metadata={"header_context": "Results"},
    )

    dumped = chunk.model_dump()

    assert dumped["chunk_type"] == "table"
    assert dumped["location"]["page_start"] == 4
    assert dumped["artifact_refs"][0]["storage_ref"] == "s3://paperintel/table-1.png"
    assert dumped["metadata"]["header_context"] == "Results"


def test_chunk_rejects_blank_text_and_blank_paper_id():
    with pytest.raises(ValidationError):
        PaperChunk(
            paper_id="2310.06825",
            chunk_index=0,
            text=" ",
            source=ChunkSource(paper_id="2310.06825"),
        )

    with pytest.raises(ValidationError):
        ChunkSource(paper_id=" ")


def test_search_query_rejects_blank_query_and_non_positive_limit():
    with pytest.raises(ValidationError):
        ChunkSearchQuery(query=" ")

    with pytest.raises(ValidationError):
        ChunkSearchQuery(query="retrieval", limit=0)
