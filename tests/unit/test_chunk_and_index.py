from agents.chunk_and_index import chunk_and_index_node
from models.retrieval import ChunkSource, PaperChunk
from models.schemas import PaperMetadata, PaperSlot
from services.chunking import ChunkingResult


class FakeChunkingService:
    def __init__(self, result: ChunkingResult | None = None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.inputs = []

    def chunk_paper(self, input):
        self.inputs.append(input)
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeRetrievalLayer:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc
        self.upserted = []

    def upsert_chunks(self, chunks):
        if self.exc is not None:
            raise self.exc
        self.upserted.extend(chunks)


def _metadata() -> PaperMetadata:
    return PaperMetadata(
        title="Retrieval Paper",
        authors=["A. Researcher"],
        arxiv_id="2310.06825",
        published_date="2023-10-01",
        abstract="Abstract.",
        categories=["cs.CL"],
    )


def _chunk() -> PaperChunk:
    return PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="Retrieval chunk.",
        source=ChunkSource(paper_id="2310.06825", session_id="session-1"),
    )


def _state() -> dict:
    return {
        "papers": [
            PaperSlot(
                paper_index=0,
                input_url="https://arxiv.org/abs/2310.06825",
                metadata=_metadata(),
                completed=True,
            )
        ],
        "metadata": _metadata(),
        "raw_text": "Full paper text.",
        "pdf_path": "/tmp/paper.pdf",
        "text_by_page": {1: "Full paper text."},
        "method_extraction": object(),
        "benchmarks": [object()],
        "production_readiness": object(),
        "engineer_report": object(),
        "full_markdown_report": "# Report",
        "ingestion_provenance": {"text_source": "pdf"},
        "confidence_scores": {"extraction": 0.9},
        "needs_human_review": False,
        "human_review_reason": None,
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
    }


def _config(chunking_service, retrieval_layer) -> dict:
    return {
        "configurable": {
            "session_id": "session-1",
            "chunking_service": chunking_service,
            "retrieval_layer": retrieval_layer,
        }
    }


def test_chunk_and_index_success_upserts_chunks_and_resets_scratch():
    chunk = _chunk()
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[chunk]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(_state(), _config(chunking, retrieval))

    assert result["processing_stage"] == "chunk_and_index"
    assert result["chunk_indexing_status"] == "success"
    assert result["chunk_count"] == 1
    assert result["chunk_indexing_error"] is None
    assert retrieval.upserted == [chunk]
    assert chunking.inputs[0].session_id == "session-1"
    assert chunking.inputs[0].input_url == "https://arxiv.org/abs/2310.06825"
    assert result["metadata"] is None
    assert result["raw_text"] is None
    assert result["text_by_page"] is None


def test_chunk_and_index_skips_when_chunking_returns_no_chunks():
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(_state(), _config(chunking, retrieval))

    assert result["chunk_indexing_status"] == "skipped"
    assert result["chunk_count"] == 0
    assert retrieval.upserted == []
    assert result["raw_text"] is None


def test_chunk_and_index_failure_is_non_fatal_and_resets_scratch():
    chunk = _chunk()
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[chunk]))
    retrieval = FakeRetrievalLayer(exc=RuntimeError("qdrant unavailable"))

    result = chunk_and_index_node(_state(), _config(chunking, retrieval))

    assert result["processing_stage"] == "chunk_and_index"
    assert result["chunk_indexing_status"] == "failed"
    assert "qdrant unavailable" in result["chunk_indexing_error"]
    assert result["raw_text"] is None
    assert len(result["errors"]) == 1
    assert result["errors"][0].severity == "warning"
    assert result["errors"][0].recoverable is True


def test_chunk_and_index_missing_finalized_slot_is_non_fatal_failure():
    chunking = FakeChunkingService()
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node({"papers": []}, _config(chunking, retrieval))

    assert result["processing_stage"] == "chunk_and_index"
    assert result["chunk_indexing_status"] == "failed"
    assert result["chunk_count"] == 0
    assert "requires a finalized PaperSlot" in result["chunk_indexing_error"]
    assert result["errors"][0].severity == "warning"
