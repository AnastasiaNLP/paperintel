from models.schemas import PaperMetadata
from services.chunking import ChunkingConfig, ChunkingInput, ChunkingService, resolve_paper_id


def _metadata(arxiv_id: str = "2310.06825v2") -> PaperMetadata:
    return PaperMetadata(
        title="Retrieval Paper",
        authors=["A. Researcher"],
        arxiv_id=arxiv_id,
        published_date="2023-10-01",
        abstract="This paper studies retrieval for engineering evidence.",
        categories=["cs.CL"],
    )


def test_resolve_paper_id_uses_versionless_arxiv_id():
    result = resolve_paper_id(ChunkingInput(metadata=_metadata()))

    assert result == "2310.06825"


def test_chunk_paper_emits_abstract_and_text_chunks_with_deterministic_ids():
    service = ChunkingService(
        ChunkingConfig(target_chars=80, overlap_chars=10, min_chunk_chars=20)
    )

    result = service.chunk_paper(
        ChunkingInput(
            metadata=_metadata(),
            raw_text="Introduction\n\n" + "Retrieval evidence is useful. " * 12,
            session_id="session-1",
            paper_index=0,
            input_url="https://arxiv.org/abs/2310.06825",
        )
    )

    assert result.paper_id == "2310.06825"
    assert result.skipped_reason is None
    assert result.chunks[0].id == "2310.06825:chunk:0"
    assert result.chunks[0].chunk_type == "abstract"
    assert result.chunks[1].id == "2310.06825:chunk:1"
    assert result.chunks[1].source.session_id == "session-1"
    assert result.chunks[1].source.input_url == "https://arxiv.org/abs/2310.06825"


def test_chunk_paper_uses_text_by_page_for_page_locations():
    service = ChunkingService(
        ChunkingConfig(
            target_chars=50,
            overlap_chars=5,
            min_chunk_chars=20,
            include_abstract=False,
        )
    )

    result = service.chunk_paper(
        ChunkingInput(
            metadata=_metadata("2401.00001"),
            text_by_page={
                2: "Results\n\n" + "Dense retrieval improves citations. " * 4,
                1: "Introduction\n\n" + "We study chunking. " * 4,
            },
        )
    )

    assert [chunk.location.page_start for chunk in result.chunks[:2]] == [1, 1]
    assert result.chunks[0].location.section_title == "Introduction"
    assert any(chunk.location.page_start == 2 for chunk in result.chunks)


def test_chunk_overlap_repeats_tail_context_between_adjacent_chunks():
    service = ChunkingService(
        ChunkingConfig(
            target_chars=40,
            overlap_chars=10,
            min_chunk_chars=20,
            include_abstract=False,
        )
    )

    result = service.chunk_paper(
        ChunkingInput(
            metadata=_metadata("2401.00002"),
            raw_text="abcdefghijklmnopqrstuvwxyz " * 6,
        )
    )

    assert len(result.chunks) > 1
    assert result.chunks[1].location.char_start < result.chunks[0].location.char_end


def test_chunk_paper_reports_skip_when_no_text_or_abstract_available():
    metadata = _metadata("2401.00003").model_copy(update={"abstract": ""})
    service = ChunkingService()

    result = service.chunk_paper(ChunkingInput(metadata=metadata))

    assert result.chunks == []
    assert result.skipped_reason == "no_text_available"
