import os
import logging
import time
import uuid

import pytest


REQUIRED_ENV = (
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGCHAIN_API_KEY",
)

QA_INTENTS = {"qa_factual", "qa_math", "qa_comparison", "qa_followup"}
PAPER_URL = "https://arxiv.org/abs/1706.03762"
QUESTION = "What is the main contribution of this paper?"


def _missing_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_missing_env()),
        reason=(
            "PAPERINTEL_TEST_DATABASE_URL, PAPERINTEL_QDRANT_TEST_URL, "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, and LANGCHAIN_API_KEY are required"
        ),
    ),
]


def test_qa_conversation_live_end_to_end():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_QA_LOG %(levelname)s %(name)s %(message)s",
        force=True,
    )

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import select

    from api.chat_handler import ChatHandler
    from graph import build_graph
    from graph_conversation import build_conversation_graph
    from services.embeddings import OpenAIEmbeddingProvider
    from services.qdrant_store import QdrantChunkStore
    from services.retrieval_layer import PostgresQdrantRetrievalLayer
    from storage.db import make_engine, make_session_factory
    from storage.mappers import orm_to_agent_run
    from storage.models import AgentRunORM
    from storage.repositories import (
        PostgresAgentRunPersistence,
        PostgresPaperChunkRepository,
        PostgresSessionStore,
        clear_foundation_tables,
    )

    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]
    qdrant_url = os.environ["PAPERINTEL_QDRANT_TEST_URL"]
    collection = f"paper_chunks_qa_live_{run_id}"

    print(f"LIVE_QA_RUN_ID={run_id}", flush=True)
    print(f"LIVE_QA_QDRANT_COLLECTION={collection}", flush=True)

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

        retrieval_layer = PostgresQdrantRetrievalLayer(
            chunk_repository=PostgresPaperChunkRepository(session_factory),
            vector_store=vector_store,
            embedding_provider=OpenAIEmbeddingProvider(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=60.0,
            ),
        )
        agent_run_persistence = PostgresAgentRunPersistence(session_factory)
        handler = ChatHandler(
            store=PostgresSessionStore(session_factory),
            conversation_runner=build_conversation_graph(),
            analysis_runner=build_graph().compile(),
            agent_run_persistence=agent_run_persistence,
            retrieval_layer=retrieval_layer,
        )

        session = handler.create_session(persona="engineer")
        print(f"LIVE_QA_SESSION_ID={session.id}", flush=True)
        print(f"LIVE_QA_PAPER_URL={PAPER_URL}", flush=True)

        analysis_started = time.monotonic()
        analysis_result = handler.handle_message(session.id, PAPER_URL)
        analysis_elapsed = time.monotonic() - analysis_started

        loaded_session = handler.store.require_session(session.id)
        active_paper_ids = list(loaded_session.active_paper_ids)
        print(f"LIVE_QA_ANALYSIS_SECONDS={analysis_elapsed:.1f}", flush=True)
        print(f"LIVE_QA_ANALYSIS_PHASE={analysis_result.phase}", flush=True)
        print(f"LIVE_QA_ACTIVE_PAPER_IDS={','.join(active_paper_ids)}", flush=True)

        assert analysis_result.intent == "analyze_paper"
        assert active_paper_ids

        qa_started = time.monotonic()
        qa_result = handler.handle_message(session.id, QUESTION)
        qa_elapsed = time.monotonic() - qa_started

        print(f"LIVE_QA_QUESTION_SECONDS={qa_elapsed:.1f}", flush=True)
        print(f"LIVE_QA_INTENT={qa_result.intent}", flush=True)
        print(f"LIVE_QA_RESPONSE_CHARS={len(qa_result.response_text)}", flush=True)
        print(f"LIVE_QA_CITATION_COUNT={len(qa_result.citations)}", flush=True)
        print(
            "LIVE_QA_REFERENCED_PAPER_IDS="
            f"{','.join(qa_result.referenced_paper_ids)}",
            flush=True,
        )

        assert qa_result.intent in QA_INTENTS
        assert len(qa_result.response_text) > 50
        assert qa_result.citations
        assert any(citation.paper_id in active_paper_ids for citation in qa_result.citations)
        assert any(paper_id in active_paper_ids for paper_id in qa_result.referenced_paper_ids)

        runs = _list_agent_runs(session_factory, session.id, orm_to_agent_run, AgentRunORM, select)
        agent_names = [run.agent_name for run in runs]
        failed_runs = [run for run in runs if run.status == "failed"]
        print(f"LIVE_QA_AGENT_RUN_COUNT={len(runs)}", flush=True)
        print(f"LIVE_QA_AGENT_RUNS={','.join(agent_names)}", flush=True)
        print(f"LIVE_QA_FAILED_RUNS={len(failed_runs)}", flush=True)

        expected_agents = {
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
            print("LIVE_QA_QDRANT_CLEANUP=success", flush=True)
        except Exception as exc:
            print(f"LIVE_QA_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

        try:
            with session_factory() as db:
                clear_foundation_tables(db)
            print("LIVE_QA_POSTGRES_CLEANUP=success", flush=True)
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
