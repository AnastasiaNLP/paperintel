from api.chat_handler import (
    AnalysisRunner,
    ChatHandler,
    ConversationRunner,
    DiscoveryRunner,
)
from services.arxiv_search_provider import ArxivSearchProvider
from services.embeddings import OpenAIEmbeddingProvider
from services.health import HealthChecker
from services.paperintel_service import PaperIntelService
from services.qdrant_store import QdrantChunkStore
from services.retrieval_layer import RetrievalLayer
from services.retrieval_layer import PostgresQdrantRetrievalLayer
from services.searcher import Searcher
from services.selection_parser import SelectionHandler
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresAgentRunPersistence,
    PostgresPaperChunkRepository,
    PostgresSearchCandidateRepository,
    PostgresSessionStore,
)


def create_chat_handler(
    *,
    database_url: str,
    conversation_runner: ConversationRunner,
    analysis_runner: AnalysisRunner | None = None,
    discovery_runner: DiscoveryRunner | None = None,
    retrieval_layer: RetrievalLayer | None = None,
) -> ChatHandler:
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    session_store = PostgresSessionStore(session_factory)
    candidate_repository = PostgresSearchCandidateRepository(session_factory)
    searcher = (
        Searcher(
            provider=ArxivSearchProvider(),
            candidate_repository=candidate_repository,
        )
        if discovery_runner is not None
        else None
    )
    return ChatHandler(
        store=session_store,
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        discovery_runner=discovery_runner,
        agent_run_persistence=PostgresAgentRunPersistence(session_factory),
        retrieval_layer=retrieval_layer,
        searcher=searcher,
        selection_handler=SelectionHandler(
            session_store=session_store,
            candidate_repository=candidate_repository,
        ),
    )


def create_paperintel_service(
    *,
    database_url: str | None = None,
    conversation_runner: ConversationRunner | None = None,
    analysis_runner: AnalysisRunner | None = None,
    discovery_runner: DiscoveryRunner | None = None,
    retrieval_layer: RetrievalLayer | None = None,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
    enable_health_checks: bool = True,
) -> PaperIntelService:
    settings = None
    if database_url is None or retrieval_layer is None or enable_health_checks:
        from config.settings import settings as loaded_settings

        settings = loaded_settings

    resolved_database_url = database_url or settings.postgres_url
    engine = make_engine(resolved_database_url)
    session_factory = make_session_factory(engine)

    vector_store = None
    if retrieval_layer is None:
        vector_store = QdrantChunkStore.from_url(
            url=qdrant_url or settings.qdrant_url,
            collection_name=qdrant_collection or settings.qdrant_collection,
            timeout=settings.qdrant_timeout,
        )
        retrieval_layer = PostgresQdrantRetrievalLayer(
            chunk_repository=PostgresPaperChunkRepository(session_factory),
            vector_store=vector_store,
            embedding_provider=OpenAIEmbeddingProvider(api_key=settings.openai_api_key),
        )
    elif hasattr(retrieval_layer, "vector_store"):
        vector_store = getattr(retrieval_layer, "vector_store")

    if conversation_runner is None:
        from graph_conversation import build_conversation_graph

        conversation_runner = build_conversation_graph()
    if analysis_runner is None:
        from graph import build_graph

        analysis_runner = build_graph().compile()
    candidate_repository = PostgresSearchCandidateRepository(session_factory)
    if discovery_runner is None:
        from graph_discovery import build_discovery_graph

        discovery_runner = build_discovery_graph()

    searcher = Searcher(
        provider=ArxivSearchProvider(),
        candidate_repository=candidate_repository,
    )
    session_store = PostgresSessionStore(session_factory)

    handler = ChatHandler(
        store=session_store,
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        discovery_runner=discovery_runner,
        agent_run_persistence=PostgresAgentRunPersistence(session_factory),
        retrieval_layer=retrieval_layer,
        searcher=searcher,
        selection_handler=SelectionHandler(
            session_store=session_store,
            candidate_repository=candidate_repository,
        ),
    )

    health_checker = None
    if enable_health_checks:
        health_checker = HealthChecker(
            session_factory=session_factory,
            qdrant_store=vector_store,
            settings=settings,
        )

    return PaperIntelService(handler=handler, health_checker=health_checker)
