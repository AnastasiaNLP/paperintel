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
QA_QUESTION = "What is the main contribution of the selected paper?"
QA_INTENTS = {"qa_factual", "qa_math", "qa_comparison", "qa_followup"}


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


def test_discovery_to_qa_live_end_to_end():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_DISCOVERY_QA_LOG %(levelname)s %(name)s %(message)s",
        force=True,
    )

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import select

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
    from storage.mappers import orm_to_agent_run
    from storage.models import AgentRunORM
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
    collection = f"paper_chunks_discovery_qa_live_{run_id}"

    print(f"LIVE_DISCOVERY_QA_RUN_ID={run_id}", flush=True)
    print(f"LIVE_DISCOVERY_QA_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_DISCOVERY_QA_QUERY={DISCOVERY_QUERY}", flush=True)

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
        agent_run_persistence = PostgresAgentRunPersistence(session_factory)
        searcher = Searcher(
            provider=ArxivSearchProvider(),
            candidate_repository=candidate_repository,
        )
        handler = ChatHandler(
            store=session_store,
            conversation_runner=build_conversation_graph(),
            analysis_runner=build_graph().compile(),
            discovery_runner=build_discovery_graph(),
            agent_run_persistence=agent_run_persistence,
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
        print(f"LIVE_DISCOVERY_QA_SESSION_ID={session.id}", flush=True)

        discovery_started = time.monotonic()
        discovery_result = service.discover_papers(session.id, DISCOVERY_QUERY)
        discovery_elapsed = time.monotonic() - discovery_started

        candidates = candidate_repository.list_latest_for_session(session.id)
        search_warnings = list(discovery_result.search_warnings)
        print(f"LIVE_DISCOVERY_QA_DISCOVERY_SECONDS={discovery_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_QA_DISCOVERY_PHASE={discovery_result.phase}", flush=True)
        print(f"LIVE_DISCOVERY_QA_CANDIDATE_COUNT={len(candidates)}", flush=True)
        print(f"LIVE_DISCOVERY_QA_WARNINGS={';'.join(search_warnings)}", flush=True)

        assert discovery_result.intent == "discover"
        assert discovery_result.phase == "selection"
        assert {"research_strategist", "selection_advisor"}.issubset(
            {run.agent_name for run in discovery_result.agent_runs}
        )
        if not candidates:
            if search_warnings and all("HTTP 429" in warning for warning in search_warnings):
                pytest.skip(
                    f"arXiv rate limited all discovery queries ({len(search_warnings)} queries). "
                    "Wait a few minutes and retry."
                )
            pytest.skip(f"arXiv returned no candidates for topic: {DISCOVERY_QUERY}")

        selection_message = f"use {candidates[0].display_rank}"
        print(f"LIVE_DISCOVERY_QA_SELECTION_MESSAGE={selection_message}", flush=True)
        selection_started = time.monotonic()
        selection_result = service.select_papers(session.id, selection_message)
        selection_elapsed = time.monotonic() - selection_started

        selected_session = service.get_session(session.id)
        selected_ids = list(selected_session.selected_candidate_ids)
        print(f"LIVE_DISCOVERY_QA_SELECTION_SECONDS={selection_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_QA_SELECTED_IDS={','.join(selected_ids)}", flush=True)

        assert selection_result.intent == "select_papers"
        assert selection_result.phase == "idle"
        assert len(selected_ids) == 1

        time.sleep(5)
        analysis_started = time.monotonic()
        analysis_result = service.analyze_selected_papers(session.id)
        analysis_elapsed = time.monotonic() - analysis_started

        analyzed_candidates = candidate_repository.get_many_by_ids(selected_ids)
        loaded_session = service.get_session(session.id)
        active_paper_ids = list(loaded_session.active_paper_ids)
        print(f"LIVE_DISCOVERY_QA_ANALYSIS_SECONDS={analysis_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_QA_ANALYSIS_PHASE={analysis_result.phase}", flush=True)
        print(f"LIVE_DISCOVERY_QA_ACTIVE_PAPER_IDS={','.join(active_paper_ids)}", flush=True)
        print(
            "LIVE_DISCOVERY_QA_SELECTED_STATUSES="
            f"{','.join(candidate.status for candidate in analyzed_candidates)}",
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
        assert active_paper_ids
        assert analyzed_candidates
        assert all(candidate.status == "analyzed" for candidate in analyzed_candidates)

        qa_started = time.monotonic()
        qa_result = service.ask_question(session.id, QA_QUESTION)
        qa_elapsed = time.monotonic() - qa_started

        print(f"LIVE_DISCOVERY_QA_QUESTION_SECONDS={qa_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_QA_INTENT={qa_result.intent}", flush=True)
        print(f"LIVE_DISCOVERY_QA_RESPONSE_CHARS={len(qa_result.response_text)}", flush=True)
        print(f"LIVE_DISCOVERY_QA_CITATION_COUNT={len(qa_result.citations)}", flush=True)
        print(
            "LIVE_DISCOVERY_QA_REFERENCED_PAPER_IDS="
            f"{','.join(qa_result.referenced_paper_ids)}",
            flush=True,
        )

        assert qa_result.intent in QA_INTENTS
        assert len(qa_result.response_text) > 50
        assert qa_result.citations
        assert any(citation.paper_id in active_paper_ids for citation in qa_result.citations)

        runs = _list_agent_runs(session_factory, session.id, orm_to_agent_run, AgentRunORM, select)
        agent_names = [run.agent_name for run in runs]
        failed_runs = [run for run in runs if run.status == "failed"]
        print(f"LIVE_DISCOVERY_QA_AGENT_RUN_COUNT={len(runs)}", flush=True)
        print(f"LIVE_DISCOVERY_QA_AGENT_RUNS={','.join(agent_names)}", flush=True)
        print(f"LIVE_DISCOVERY_QA_FAILED_RUNS={len(failed_runs)}", flush=True)

        expected_agents = {
            "research_strategist",
            "selection_advisor",
            "report",
            "evidence_critic",
            "intent_router",
            "retrieval_planner",
            "answer_agent",
            "citation_critic",
        }
        assert expected_agents.issubset(set(agent_names))
        assert not failed_runs
    finally:
        try:
            vector_store.client.delete_collection(collection_name=collection)
            print("LIVE_DISCOVERY_QA_QDRANT_CLEANUP=success", flush=True)
        except Exception as exc:
            print(
                f"LIVE_DISCOVERY_QA_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}",
                flush=True,
            )

        try:
            with session_factory() as db:
                clear_foundation_tables(db)
            print("LIVE_DISCOVERY_QA_POSTGRES_CLEANUP=success", flush=True)
        finally:
            engine.dispose()
def _list_agent_runs(
    session_factory,
    session_id,
    orm_to_agent_run,
    agent_run_orm,
    select,
):
    with session_factory() as db:
        rows = (
            db.execute(
                select(agent_run_orm)
                .where(agent_run_orm.session_id == session_id)
                .order_by(agent_run_orm.started_at.asc())
            )
            .scalars()
            .all()
        )
    return [orm_to_agent_run(row) for row in rows]


def _is_external_arxiv_metadata_failure(result) -> bool:
    messages = [error.message for error in result.errors]
    messages.append(result.response_text)
    return any("arXiv metadata failed" in message for message in messages)
