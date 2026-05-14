from agents.chunk_and_index import chunk_and_index_node
from api.in_memory_session_store import InMemorySessionStore
from models.retrieval import ChunkSource, PaperChunk
from models.schemas import PaperMetadata, PaperSlot
from services.chunking import ChunkingResult


class FakeChunkingService:
    def __init__(self, result: ChunkingResult | None = None):
        self.result = result

    def chunk_paper(self, input):
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
        "text_by_page": {1: "Full paper text."},
    }


def _config(chunking_service, retrieval_layer, session_store=None, session_id="session-1"):
    configurable = {
        "session_id": session_id,
        "chunking_service": chunking_service,
        "retrieval_layer": retrieval_layer,
    }
    if session_store is not None:
        configurable["session_store"] = session_store
    return {"configurable": configurable}


def test_chunk_and_index_adds_paper_to_active_on_success():
    store = InMemorySessionStore()
    session = store.create_session()
    chunk = _chunk().model_copy(
        update={"source": ChunkSource(paper_id="2310.06825", session_id=session.id)}
    )
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[chunk]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(
        _state(),
        _config(chunking, retrieval, session_store=store, session_id=session.id),
    )

    assert result["chunk_indexing_status"] == "success"
    assert store.require_session(session.id).active_paper_ids == ["2310.06825"]


def test_chunk_and_index_does_not_add_paper_on_failure():
    store = InMemorySessionStore()
    session = store.create_session()
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[_chunk()]))
    retrieval = FakeRetrievalLayer(exc=RuntimeError("qdrant unavailable"))

    result = chunk_and_index_node(
        _state(),
        _config(chunking, retrieval, session_store=store, session_id=session.id),
    )

    assert result["chunk_indexing_status"] == "failed"
    assert store.require_session(session.id).active_paper_ids == []


def test_chunk_and_index_does_not_add_paper_on_skipped():
    store = InMemorySessionStore()
    session = store.create_session()
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(
        _state(),
        _config(chunking, retrieval, session_store=store, session_id=session.id),
    )

    assert result["chunk_indexing_status"] == "skipped"
    assert store.require_session(session.id).active_paper_ids == []


def test_chunk_and_index_non_fatal_when_session_store_missing():
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[_chunk()]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(_state(), _config(chunking, retrieval))

    assert result["chunk_indexing_status"] == "success"
    assert result["errors"] == []


def test_chunk_and_index_success_survives_active_paper_update_failure():
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[_chunk()]))
    retrieval = FakeRetrievalLayer()
    store = InMemorySessionStore()

    result = chunk_and_index_node(
        _state(),
        _config(chunking, retrieval, session_store=store, session_id="missing"),
    )

    assert result["chunk_indexing_status"] == "success"
    assert result["errors"][0].node == "chunk_and_index"
    assert result["errors"][0].severity == "error"
    assert result["errors"][0].recoverable is False
    assert "could not mark paper as active" in result["errors"][0].message


class BrokenSessionStore:
    def add_active_paper(self, session_id, paper_id):
        raise RuntimeError("db temporarily unavailable")


def test_chunk_and_index_marks_transient_active_paper_update_failure_recoverable():
    chunking = FakeChunkingService(ChunkingResult(paper_id="2310.06825", chunks=[_chunk()]))
    retrieval = FakeRetrievalLayer()

    result = chunk_and_index_node(
        _state(),
        _config(
            chunking,
            retrieval,
            session_store=BrokenSessionStore(),
            session_id="session-1",
        ),
    )

    assert result["chunk_indexing_status"] == "success"
    assert result["errors"][0].severity == "warning"
    assert result["errors"][0].recoverable is True
