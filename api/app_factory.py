from api.chat_handler import (
    AnalysisRunner,
    ChatHandler,
    ConversationRunner,
    DiscoveryRunner,
)
from services.arxiv_search_provider import ArxivSearchProvider
from services.embeddings import OpenAIEmbeddingProvider
from services.blob_store import BlobStore, S3BlobStore
from services.health import HealthChecker
from services.paperintel_service import PaperChunkRepository, PaperIntelService
from services.qdrant_store import QdrantChunkStore
from services.retrieval_layer import RetrievalLayer
from services.retrieval_layer import PostgresQdrantRetrievalLayer
from services.searcher import Searcher
from services.selected_candidate_resolver import SelectedCandidateResolver
from services.selection_parser import SelectionHandler
from storage.db import make_engine, make_session_factory
from storage.repositories import (
    PostgresAgentRunPersistence,
    PostgresArxivMetadataCacheRepository,
    PostgresBlobArtifactRepository,
    PostgresPaperChunkRepository,
    PostgresPaperWorkspaceRepository,
    PostgresPdfUploadRepository,
    PostgresSearchCandidateRepository,
    PostgresSessionStore,
    PostgresWorkflowJobRepository,
)
from tools.arxiv_client import configure_metadata_cache


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
    configure_metadata_cache(PostgresArxivMetadataCacheRepository(session_factory))
    session_store = PostgresSessionStore(session_factory)
    candidate_repository = PostgresSearchCandidateRepository(session_factory)
    artifact_repository = PostgresPaperWorkspaceRepository(session_factory)
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
        artifact_repository=artifact_repository,
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
    paper_chunk_repository: PaperChunkRepository | None = None,
    blob_store: BlobStore | None = None,
    enable_blob_storage: bool | None = None,
) -> PaperIntelService:
    settings = None
    if (
        database_url is None
        or retrieval_layer is None
        or enable_health_checks
        or (blob_store is None and enable_blob_storage is not False)
    ):
        from config.settings import settings as loaded_settings

        settings = loaded_settings

    resolved_database_url = database_url or settings.postgres_url
    engine = make_engine(resolved_database_url)
    session_factory = make_session_factory(engine)
    configure_metadata_cache(PostgresArxivMetadataCacheRepository(session_factory))
    blob_storage_required = _blob_storage_required(
        blob_store=blob_store,
        enable_blob_storage=enable_blob_storage,
        settings=settings,
    )
    resolved_blob_store = _resolve_blob_store(
        blob_store=blob_store,
        enable_blob_storage=enable_blob_storage,
        settings=settings,
    )

    vector_store = None
    if retrieval_layer is None:
        if paper_chunk_repository is None:
            paper_chunk_repository = PostgresPaperChunkRepository(session_factory)
        vector_store = QdrantChunkStore.from_url(
            url=qdrant_url or settings.qdrant_url,
            collection_name=qdrant_collection or settings.qdrant_collection,
            timeout=settings.qdrant_timeout,
        )
        retrieval_layer = PostgresQdrantRetrievalLayer(
            chunk_repository=paper_chunk_repository,
            vector_store=vector_store,
            embedding_provider=OpenAIEmbeddingProvider(api_key=settings.openai_api_key),
        )
    else:
        if paper_chunk_repository is None and hasattr(retrieval_layer, "chunk_repository"):
            paper_chunk_repository = getattr(retrieval_layer, "chunk_repository")
        if paper_chunk_repository is None:
            paper_chunk_repository = PostgresPaperChunkRepository(session_factory)
        if hasattr(retrieval_layer, "vector_store"):
            vector_store = getattr(retrieval_layer, "vector_store")

    if conversation_runner is None:
        from graph_conversation import build_conversation_graph

        conversation_runner = build_conversation_graph()
    if analysis_runner is None:
        from graph import build_graph

        analysis_runner = build_graph().compile()
    candidate_repository = PostgresSearchCandidateRepository(session_factory)
    artifact_repository = PostgresPaperWorkspaceRepository(session_factory)
    workflow_job_repository = PostgresWorkflowJobRepository(session_factory)
    blob_artifact_repository = PostgresBlobArtifactRepository(session_factory)
    pdf_upload_repository = PostgresPdfUploadRepository(session_factory)
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
        artifact_repository=artifact_repository,
    )

    health_checker = None
    if enable_health_checks:
        health_checker = HealthChecker(
            session_factory=session_factory,
            qdrant_store=vector_store,
            settings=settings,
            blob_store=resolved_blob_store,
            blob_storage_required=blob_storage_required,
        )

    selected_candidate_resolver = SelectedCandidateResolver(
        session_store=session_store,
        candidate_repository=candidate_repository,
    )
    return PaperIntelService(
        handler=handler,
        health_checker=health_checker,
        selected_candidate_resolver=selected_candidate_resolver,
        candidate_repository=candidate_repository,
        artifact_repository=artifact_repository,
        workflow_job_repository=workflow_job_repository,
        paper_chunk_repository=paper_chunk_repository,
        blob_store=resolved_blob_store,
        blob_artifact_repository=(
            blob_artifact_repository if resolved_blob_store is not None else None
        ),
        pdf_upload_repository=(
            pdf_upload_repository if resolved_blob_store is not None else None
        ),
    )


def _blob_storage_required(
    *,
    blob_store: BlobStore | None,
    enable_blob_storage: bool | None,
    settings,
) -> bool:
    if blob_store is not None:
        return True
    if enable_blob_storage is not None:
        return enable_blob_storage
    return bool(settings is not None and settings.blob_storage_enabled)



def _resolve_blob_store(
    *,
    blob_store: BlobStore | None,
    enable_blob_storage: bool | None,
    settings,
) -> BlobStore | None:
    if blob_store is not None:
        blob_store.ensure_bucket()
        return blob_store
    if enable_blob_storage is False:
        return None
    if settings is None:
        return None
    if enable_blob_storage is None and not settings.blob_storage_enabled:
        return None
    resolved = S3BlobStore.from_config(
        bucket_name=settings.blob_s3_bucket,
        endpoint_url=settings.blob_s3_endpoint_url,
        region_name=settings.blob_s3_region,
        access_key_id=settings.blob_s3_access_key_id,
        secret_access_key=settings.blob_s3_secret_access_key,
    )
    resolved.ensure_bucket()
    return resolved
