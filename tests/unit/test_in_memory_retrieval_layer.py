from models.retrieval import ChunkLocation, ChunkSearchQuery, ChunkSource, PaperChunk
from services.retrieval_layer import InMemoryRetrievalLayer


def _chunk(
    *,
    chunk_id: str,
    paper_id: str,
    chunk_index: int,
    text: str,
    session_id: str | None = "session-1",
) -> PaperChunk:
    return PaperChunk(
        id=chunk_id,
        paper_id=paper_id,
        chunk_index=chunk_index,
        text=text,
        source=ChunkSource(
            paper_id=paper_id,
            session_id=session_id,
            arxiv_id=paper_id,
            title=f"Paper {paper_id}",
        ),
        location=ChunkLocation(page_start=chunk_index + 1, page_end=chunk_index + 1),
    )


def test_upsert_chunks_is_idempotent_by_chunk_id():
    layer = InMemoryRetrievalLayer()
    first = _chunk(
        chunk_id="chunk-1",
        paper_id="2310.06825",
        chunk_index=0,
        text="Retrieval augmented generation.",
    )
    second = first.model_copy(update={"text": "Updated retrieval text."})

    assert layer.upsert_chunks([first]).model_dump() == {
        "inserted": 1,
        "updated": 0,
        "skipped": 0,
    }
    assert layer.upsert_chunks([second]).model_dump() == {
        "inserted": 0,
        "updated": 1,
        "skipped": 0,
    }

    results = layer.search_chunks(ChunkSearchQuery(query="updated", limit=10))

    assert len(results) == 1
    assert results[0].chunk.text == "Updated retrieval text."


def test_search_chunks_scores_filters_limits_and_ranks_deterministically():
    layer = InMemoryRetrievalLayer()
    layer.upsert_chunks(
        [
            _chunk(
                chunk_id="chunk-1",
                paper_id="2310.06825",
                chunk_index=0,
                text="Sparse retrieval and dense retrieval are compared.",
            ),
            _chunk(
                chunk_id="chunk-2",
                paper_id="2310.06825",
                chunk_index=1,
                text="Production readiness discusses deployment cost.",
            ),
            _chunk(
                chunk_id="chunk-3",
                paper_id="2401.00001",
                chunk_index=0,
                text="Dense retrieval improves citation quality.",
                session_id="session-2",
            ),
        ]
    )

    results = layer.search_chunks(
        ChunkSearchQuery(
            query="dense retrieval citation",
            session_id="session-1",
            paper_ids=["2310.06825"],
            limit=1,
        )
    )

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].chunk.id == "chunk-1"
    assert results[0].match_reason == "lexical_token_overlap"


def test_search_chunks_returns_empty_list_for_no_matches():
    layer = InMemoryRetrievalLayer()
    layer.upsert_chunks(
        [
            _chunk(
                chunk_id="chunk-1",
                paper_id="2310.06825",
                chunk_index=0,
                text="Benchmark extraction.",
            )
        ]
    )

    assert layer.search_chunks(ChunkSearchQuery(query="unrelated")) == []


def test_assemble_evidence_builds_citations_from_results():
    layer = InMemoryRetrievalLayer()
    chunk = _chunk(
        chunk_id="chunk-1",
        paper_id="2310.06825",
        chunk_index=2,
        text="Evidence retrieval supports citations.",
    )
    layer.upsert_chunks([chunk])
    results = layer.search_chunks(ChunkSearchQuery(query="retrieval citations"))

    bundle = layer.assemble_evidence("retrieval citations", results)

    assert bundle.query == "retrieval citations"
    assert len(bundle.results) == 1
    assert bundle.citations[0].paper_id == "2310.06825"
    assert bundle.citations[0].chunk_id == "chunk-1"
    assert bundle.citations[0].page_start == 3
    assert bundle.coverage_notes == []


def test_assemble_evidence_notes_empty_coverage():
    layer = InMemoryRetrievalLayer()

    bundle = layer.assemble_evidence("missing", [])

    assert bundle.results == []
    assert bundle.citations == []
    assert bundle.coverage_notes == ["no_matching_chunks"]
