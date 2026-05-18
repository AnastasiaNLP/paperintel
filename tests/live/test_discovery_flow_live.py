import logging
import os
import time
import uuid

import pytest


REQUIRED_ENV = (
    "PAPERINTEL_TEST_DATABASE_URL",
    "ANTHROPIC_API_KEY",
)

DISCOVERY_QUERY = "Find recent papers about retrieval augmented generation"


def _missing_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_missing_env()),
        reason="PAPERINTEL_TEST_DATABASE_URL and ANTHROPIC_API_KEY are required",
    ),
]


def test_discovery_flow_live_end_to_end():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_DISCOVERY_LOG %(levelname)s %(name)s %(message)s",
        force=True,
    )

    from alembic import command
    from alembic.config import Config

    from api.chat_handler import ChatHandler
    from graph_conversation import build_conversation_graph
    from graph_discovery import build_discovery_graph
    from services.arxiv_search_provider import ArxivSearchProvider
    from services.searcher import Searcher
    from services.selection_parser import SelectionHandler
    from storage.db import make_engine, make_session_factory
    from storage.repositories import (
        PostgresAgentRunPersistence,
        PostgresSearchCandidateRepository,
        PostgresSessionStore,
        clear_foundation_tables,
    )

    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]

    print(f"LIVE_DISCOVERY_RUN_ID={run_id}", flush=True)
    print(f"LIVE_DISCOVERY_QUERY={DISCOVERY_QUERY}", flush=True)

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)

    try:
        with session_factory() as db:
            clear_foundation_tables(db)

        session_store = PostgresSessionStore(session_factory)
        candidate_repository = PostgresSearchCandidateRepository(session_factory)
        agent_run_persistence = PostgresAgentRunPersistence(session_factory)
        searcher = Searcher(
            provider=ArxivSearchProvider(),
            candidate_repository=candidate_repository,
        )
        handler = ChatHandler(
            store=session_store,
            conversation_runner=build_conversation_graph(),
            discovery_runner=build_discovery_graph(),
            agent_run_persistence=agent_run_persistence,
            searcher=searcher,
            selection_handler=SelectionHandler(
                session_store=session_store,
                candidate_repository=candidate_repository,
            ),
        )

        session = handler.create_session(persona="engineer")
        print(f"LIVE_DISCOVERY_SESSION_ID={session.id}", flush=True)

        discovery_started = time.monotonic()
        discovery_result = handler.handle_message(session.id, DISCOVERY_QUERY)
        discovery_elapsed = time.monotonic() - discovery_started

        candidates = candidate_repository.list_latest_for_session(session.id)
        agent_names = [run.agent_name for run in discovery_result.agent_runs]
        failed_runs = [
            run for run in discovery_result.agent_runs if run.status == "failed"
        ]
        search_warnings = discovery_result.search_warnings or []
        print(f"LIVE_DISCOVERY_SECONDS={discovery_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_PHASE={discovery_result.phase}", flush=True)
        print(f"LIVE_DISCOVERY_RESPONSE_CHARS={len(discovery_result.response_text)}", flush=True)
        print(f"LIVE_DISCOVERY_CANDIDATE_COUNT={len(candidates)}", flush=True)
        print(f"LIVE_DISCOVERY_WARNINGS={';'.join(search_warnings)}", flush=True)
        print(f"LIVE_DISCOVERY_AGENT_RUNS={','.join(agent_names)}", flush=True)
        print(f"LIVE_DISCOVERY_FAILED_RUNS={len(failed_runs)}", flush=True)

        assert discovery_result.intent == "discover"
        assert discovery_result.phase == "selection"
        assert len(discovery_result.response_text) > 100
        assert {"research_strategist", "selection_advisor"}.issubset(set(agent_names))
        assert not failed_runs

        if search_warnings and all("HTTP 429" in warning for warning in search_warnings):
            pytest.skip(
                f"arXiv rate limited all discovery queries ({len(search_warnings)} queries). "
                "Wait a few minutes and retry."
            )

        if len(candidates) < 2:
            pytest.skip(f"arXiv returned only {len(candidates)} candidates, need >= 2")

        assert "number" in discovery_result.response_text.lower()
        assert candidates[0].display_rank == 1
        assert [candidate.display_rank for candidate in candidates] == list(
            range(1, len(candidates) + 1)
        )

        n_select = min(2, len(candidates))
        ranks = [str(candidate.display_rank) for candidate in candidates[:n_select]]
        selection_message = "use " + " and ".join(ranks)
        print(f"LIVE_SELECTION_MESSAGE={selection_message}", flush=True)

        selection_started = time.monotonic()
        selection_result = handler.handle_message(session.id, selection_message)
        selection_elapsed = time.monotonic() - selection_started

        loaded_session = session_store.require_session(session.id)
        selected_ids = list(loaded_session.selected_candidate_ids)
        latest_candidates = candidate_repository.list_latest_for_session(session.id)
        selected_candidates = [
            candidate for candidate in latest_candidates if candidate.id in selected_ids
        ]
        print(f"LIVE_SELECTION_SECONDS={selection_elapsed:.1f}", flush=True)
        print(f"LIVE_DISCOVERY_SELECTED_IDS={','.join(selected_ids)}", flush=True)
        print(
            "LIVE_DISCOVERY_SELECTED_STATUSES="
            f"{','.join(candidate.status for candidate in selected_candidates)}",
            flush=True,
        )

        assert selection_result.intent == "select_papers"
        assert selection_result.phase == "idle"
        assert len(selected_ids) == n_select
        assert len(selected_candidates) == n_select
        assert all(candidate.status == "selected" for candidate in selected_candidates)
    finally:
        try:
            with session_factory() as db:
                clear_foundation_tables(db)
            print("LIVE_DISCOVERY_CLEANUP=success", flush=True)
        finally:
            engine.dispose()
