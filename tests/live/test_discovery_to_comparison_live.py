import logging
import os
import time
import uuid

import pytest


REQUIRED_ENV = (
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

DISCOVERY_QUERY = "Find recent papers about retrieval augmented generation"


def _missing_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_missing_env()),
        reason=(
            "PAPERINTEL_TEST_DATABASE_URL, PAPERINTEL_QDRANT_TEST_URL, "
            "ANTHROPIC_API_KEY, and OPENAI_API_KEY are required"
        ),
    ),
]


def test_discovery_to_comparison_live_end_to_end():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_DISCOVERY_COMPARISON_LOG %(levelname)s %(name)s %(message)s",
        force=True,
    )

    from alembic import command
    from alembic.config import Config

    from api.chat_handler import ChatHandler
    from graph import build_graph
    from graph_conversation import build_conversation_graph
    from graph_discovery import build_discovery_graph
    from services.arxiv_search_provider import ArxivSearchProvider
    from services.embeddings import OpenAIEmbeddingProvider
    from services.paperintel_service import PaperIntelService
    from services.qdrant_store import QdrantChunkStore
    from services.retrieval_layer import PostgresQdrantRetrievalLayer
    from services.searcher import Searcher
    from services.selected_candidate_resolver import SelectedCandidateResolver
    from services.selection_parser import SelectionHandler
    from storage.db import make_engine, make_session_factory
    from storage.repositories import (
        PostgresAgentRunPersistence,
        PostgresPaperChunkRepository,
        PostgresSearchCandidateRepository,
        PostgresSessionStore,
        clear_foundation_tables,
    )

    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]
    qdrant_url = os.environ["PAPERINTEL_QDRANT_TEST_URL"]
    collection = f"paper_chunks_discovery_comparison_live_{run_id}"

    print(f"LIVE_DISCOVERY_COMPARISON_RUN_ID={run_id}", flush=True)
    print(f"LIVE_DISCOVERY_COMPARISON_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_DISCOVERY_COMPARISON_QUERY={DISCOVERY_QUERY}", flush=True)

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    vector_store = QdrantChunkStore.from_url(
        url=qdrant_url,
        collection_name=collection,
        timeout=30.0,
    )

    try:
        with session_factory() as db:
            clear_foundation_tables(db)

        session_store = PostgresSessionStore(session_factory)
        candidate_repository = PostgresSearchCandidateRepository(session_factory)
        retrieval_layer = PostgresQdrantRetrievalLayer(
            chunk_repository=PostgresPaperChunkRepository(session_factory),
            vector_store=vector_store,
            embedding_provider=OpenAIEmbeddingProvider(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=60.0,
            ),
        )
        searcher = Searcher(
            provider=ArxivSearchProvider(),
            candidate_repository=candidate_repository,
        )
        handler = ChatHandler(
            store=session_store,
            conversation_runner=build_conversation_graph(),
            analysis_runner=build_graph().compile(),
            discovery_runner=build_discovery_graph(),
            agent_run_persistence=PostgresAgentRunPersistence(session_factory),
            retrieval_layer=retrieval_layer,
            searcher=searcher,
            selection_handler=SelectionHandler(
                session_store=session_store,
                candidate_repository=candidate_repository,
            ),
        )
        service = PaperIntelService(
            handler=handler,
            selected_candidate_resolver=SelectedCandidateResolver(
                session_store=session_store,
                candidate_repository=candidate_repository,
            ),
            candidate_repository=candidate_repository,
        )

        session = service.create_session(persona="engineer")
        print(f"LIVE_DISCOVERY_COMPARISON_SESSION_ID={session.id}", flush=True)

        discovery_started = time.monotonic()
        discovery_result = service.discover_papers(session.id, DISCOVERY_QUERY)
        discovery_elapsed = time.monotonic() - discovery_started

        candidates = candidate_repository.list_latest_for_session(session.id)
        search_warnings = list(discovery_result.search_warnings)
        print(
            f"LIVE_DISCOVERY_COMPARISON_DISCOVERY_SECONDS={discovery_elapsed:.1f}",
            flush=True,
        )
        print(f"LIVE_DISCOVERY_COMPARISON_CANDIDATE_COUNT={len(candidates)}", flush=True)
        print(
            f"LIVE_DISCOVERY_COMPARISON_WARNINGS={';'.join(search_warnings)}",
            flush=True,
        )

        assert discovery_result.intent == "discover"
        assert discovery_result.phase == "selection"
        if len(candidates) < 2:
            if search_warnings and all("HTTP 429" in warning for warning in search_warnings):
                pytest.skip(
                    f"arXiv rate limited all discovery queries ({len(search_warnings)} queries). "
                    "Wait a few minutes and retry."
                )
            pytest.skip(
                f"arXiv returned only {len(candidates)} candidates for topic: {DISCOVERY_QUERY}"
            )

        ranks = [str(candidate.display_rank) for candidate in candidates[:2]]
        selection_message = "use " + " and ".join(ranks)
        print(
            f"LIVE_DISCOVERY_COMPARISON_SELECTION_MESSAGE={selection_message}",
            flush=True,
        )
        selection_started = time.monotonic()
        selection_result = service.select_papers(session.id, selection_message)
        selection_elapsed = time.monotonic() - selection_started

        selected_session = service.get_session(session.id)
        selected_ids = list(selected_session.selected_candidate_ids)
        print(
            f"LIVE_DISCOVERY_COMPARISON_SELECTION_SECONDS={selection_elapsed:.1f}",
            flush=True,
        )
        print(
            f"LIVE_DISCOVERY_COMPARISON_SELECTED_IDS={','.join(selected_ids)}",
            flush=True,
        )

        assert selection_result.intent == "select_papers"
        assert selection_result.phase == "idle"
        assert len(selected_ids) == 2

        time.sleep(5)
        analysis_started = time.monotonic()
        analysis_result = service.analyze_selected_papers(session.id)
        analysis_elapsed = time.monotonic() - analysis_started

        analyzed_candidates = candidate_repository.get_many_by_ids(selected_ids)
        loaded_session = service.get_session(session.id)
        active_paper_ids = list(loaded_session.active_paper_ids)
        comparison_markdown = analysis_result.comparison_markdown or ""
        print(
            f"LIVE_DISCOVERY_COMPARISON_ANALYSIS_SECONDS={analysis_elapsed:.1f}",
            flush=True,
        )
        print(f"LIVE_DISCOVERY_COMPARISON_ANALYSIS_PHASE={analysis_result.phase}", flush=True)
        print(
            f"LIVE_DISCOVERY_COMPARISON_ACTIVE_PAPER_IDS={','.join(active_paper_ids)}",
            flush=True,
        )
        print(
            "LIVE_DISCOVERY_COMPARISON_SELECTED_STATUSES="
            f"{','.join(candidate.status for candidate in analyzed_candidates)}",
            flush=True,
        )
        print(
            f"LIVE_DISCOVERY_COMPARISON_MARKDOWN_CHARS={len(comparison_markdown)}",
            flush=True,
        )

        assert analysis_result.intent == "analyze_paper"
        if analysis_result.phase == "failed" and _is_external_arxiv_metadata_failure(
            analysis_result
        ):
            pytest.skip(
                "Selected paper analysis hit an external arXiv metadata failure. "
                "Retry later or choose another candidate."
            )
        assert analysis_result.phase == "qa"
        assert len(active_paper_ids) >= 2
        assert analyzed_candidates
        assert all(candidate.status == "analyzed" for candidate in analyzed_candidates)
        assert comparison_markdown
        assert "Paper Comparison" in comparison_markdown
    finally:
        try:
            vector_store.client.delete_collection(collection_name=collection)
            print("LIVE_DISCOVERY_COMPARISON_QDRANT_CLEANUP=success", flush=True)
        except Exception as exc:
            print(
                f"LIVE_DISCOVERY_COMPARISON_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}",
                flush=True,
            )

        try:
            with session_factory() as db:
                clear_foundation_tables(db)
            print("LIVE_DISCOVERY_COMPARISON_POSTGRES_CLEANUP=success", flush=True)
        finally:
            engine.dispose()


def _is_external_arxiv_metadata_failure(result) -> bool:
    messages = [error.message for error in result.errors]
    messages.append(result.response_text)
    return any("arXiv metadata failed" in message for message in messages)
