from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.report_finalize import _SCRATCH_RESET
from models.errors import ErrorCodes, make_error
from models.schemas import PaperSlot
from models.state import PaperIntelState
from services.chunking import ChunkingInput, ChunkingService
from services.embeddings import OpenAIEmbeddingProvider
from services.qdrant_store import QdrantChunkStore
from services.retrieval_layer import PostgresQdrantRetrievalLayer, RetrievalLayer
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresPaperChunkRepository


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _chunking_service(config: RunnableConfig | None) -> ChunkingService:
    service = _configurable(config).get("chunking_service")
    if service is not None:
        return service
    return ChunkingService()


def _retrieval_layer(config: RunnableConfig | None) -> RetrievalLayer:
    configurable = _configurable(config)
    layer = configurable.get("retrieval_layer")
    if layer is not None:
        return layer

    session_factory = configurable.get("session_factory")
    if session_factory is None:
        from config.settings import settings

        engine = make_engine(settings.postgres_url)
        session_factory = make_session_factory(engine)
    else:
        from config.settings import settings

    return PostgresQdrantRetrievalLayer(
        chunk_repository=PostgresPaperChunkRepository(session_factory),
        vector_store=QdrantChunkStore.from_url(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            timeout=settings.qdrant_timeout,
        ),
        embedding_provider=OpenAIEmbeddingProvider(api_key=settings.openai_api_key),
    )


def _last_finalized_slot(state: PaperIntelState | dict) -> PaperSlot | None:
    papers = state.get("papers") or []
    if not papers:
        return None
    slot = papers[-1]
    return slot if isinstance(slot, PaperSlot) else None


def _reset_with_status(
    *,
    status: str,
    chunk_count: int,
    error: str | None = None,
    errors: list | None = None,
) -> dict:
    return {
        "processing_stage": "chunk_and_index",
        "chunk_indexing_status": status,
        "chunk_indexing_error": error,
        "chunk_count": chunk_count,
        "errors": errors or [],
        **_SCRATCH_RESET,
    }


def chunk_and_index_node(
    state: PaperIntelState | dict,
    config: RunnableConfig | None = None,
) -> dict:
    slot = _last_finalized_slot(state)
    if slot is None:
        message = "chunk_and_index requires a finalized PaperSlot"
        return _reset_with_status(
            status="failed",
            chunk_count=0,
            error=message,
            errors=[
                make_error(
                    ErrorCodes.WARNING,
                    message,
                    node="chunk_and_index",
                    severity="warning",
                    recoverable=True,
                )
            ],
        )

    try:
        result = _chunking_service(config).chunk_paper(
            ChunkingInput(
                metadata=state.get("metadata"),
                raw_text=state.get("raw_text"),
                text_by_page=state.get("text_by_page"),
                session_id=_configurable(config).get("session_id"),
                paper_index=slot.paper_index,
                input_url=slot.input_url,
            )
        )

        if not result.chunks:
            return _reset_with_status(status="skipped", chunk_count=0)

        _retrieval_layer(config).upsert_chunks(result.chunks)
        return _reset_with_status(status="success", chunk_count=len(result.chunks))
    except Exception as exc:
        message = f"chunk_and_index failed for paper {slot.paper_index}: {exc}"
        return _reset_with_status(
            status="failed",
            chunk_count=0,
            error=message,
            errors=[
                make_error(
                    ErrorCodes.WARNING,
                    message,
                    node="chunk_and_index",
                    severity="warning",
                    recoverable=True,
                    exception_type=type(exc).__name__,
                )
            ],
        )
