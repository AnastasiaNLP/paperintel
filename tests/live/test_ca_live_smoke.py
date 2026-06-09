import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest


REQUIRED_ENV = (
    "PAPERINTEL_RUN_LIVE_CA_SMOKE",
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
PAPER_IDS = ["1706.03762", "2005.11401"]
PDF_DIR = Path("~/Desktop/pdfs").expanduser()


def _missing_env() -> list[str]:
    missing = []
    for name in REQUIRED_ENV:
        value = os.environ.get(name)
        if name == "PAPERINTEL_RUN_LIVE_CA_SMOKE":
            if value != "1":
                missing.append(name)
        elif not value:
            missing.append(name)
    return missing


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_missing_env()),
        reason=(
            "PAPERINTEL_RUN_LIVE_CA_SMOKE=1, PAPERINTEL_TEST_DATABASE_URL, "
            "PAPERINTEL_QDRANT_TEST_URL, ANTHROPIC_API_KEY, and OPENAI_API_KEY "
            "are required"
        ),
    ),
]


@dataclass(frozen=True)
class LiveStack:
    service: object
    session_factory: object
    vector_store: object
    engine: object
    collection: str


def test_ca_live_smoke_rest_mcp_and_persistence():
    _require_local_pdfs()
    if os.environ.get("PAPERINTEL_LIVE_TRACE") != "1":
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_CA_SMOKE %(levelname)s %(name)s %(message)s",
        force=True,
    )

    stack = _build_live_stack()
    app = _create_rest_app(stack.service)

    try:
        session_a = stack.service.create_session(persona="engineer")
        print(f"LIVE_CA_SESSION_A={session_a.id}", flush=True)
        _analyze_local_pair(stack.service, session_a.id, label="A")
        _assert_workspace_count(stack.service, session_a.id, expected=2, label="A")
        assert _latest_comparison_or_none(stack.service, session_a.id) is None

        response_a = _post_json(app, "POST", f"/sessions/{session_a.id}/synthesize")
        payload_a = response_a.json()
        print(
            f"LIVE_CA_SESSION_A_SYNTH_RESPONSE_CHARS={len(payload_a.get('response_text', ''))}",
            flush=True,
        )
        assert response_a.status_code == 200, payload_a
        _assert_synthesis_response(payload_a)
        assert _latest_comparison_or_none(stack.service, session_a.id) is None
        runs_a = _list_agent_runs(stack.session_factory, session_a.id)
        _print_runs("A", runs_a)
        synthesis_a = _require_run(runs_a, "synthesis_agent")
        _assert_completed_run(synthesis_a, output_ref="synthesis_report")
        _assert_comparison_count(stack.session_factory, session_a.id, expected=0)

        session_b = stack.service.create_session(persona="techlead")
        print(f"LIVE_CA_SESSION_B={session_b.id}", flush=True)
        _analyze_local_pair(stack.service, session_b.id, label="B")
        _assert_workspace_count(stack.service, session_b.id, expected=2, label="B")

        compare_response = _post_json(
            app,
            "POST",
            f"/sessions/{session_b.id}/compare",
            json={"paper_ids": PAPER_IDS, "prompt": "Prefer production readiness."},
        )
        compare_payload = compare_response.json()
        print(
            f"LIVE_CA_SESSION_B_COMPARE_RESPONSE_CHARS={len(compare_payload.get('comparison_markdown', ''))}",
            flush=True,
        )
        assert compare_response.status_code == 200, compare_payload
        assert compare_payload["paper_ids"] == PAPER_IDS
        assert compare_payload["comparison_report_json"]["producer"] == "comparison_analyst"
        assert compare_payload["comparison_markdown"]

        latest_response = _post_json(app, "GET", f"/sessions/{session_b.id}/comparison")
        latest_payload = latest_response.json()
        assert latest_response.status_code == 200, latest_payload
        assert latest_payload["id"] == compare_payload["id"]

        synth_response_b = _post_json(app, "POST", f"/sessions/{session_b.id}/synthesize")
        synth_payload_b = synth_response_b.json()
        print(
            f"LIVE_CA_SESSION_B_SYNTH_RESPONSE_CHARS={len(synth_payload_b.get('response_text', ''))}",
            flush=True,
        )
        assert synth_response_b.status_code == 200, synth_payload_b
        _assert_synthesis_response(synth_payload_b)

        runs_b = _list_agent_runs(stack.session_factory, session_b.id)
        _print_runs("B", runs_b)
        comparison_run = _require_run(runs_b, "comparison_analyst")
        synthesis_b = _require_run(runs_b, "synthesis_agent")
        _assert_completed_run(comparison_run, output_ref="comparison_report")
        _assert_completed_run(synthesis_b, output_ref="synthesis_report")
        assert f"comparison_artifact:{compare_payload['id']}" in synthesis_b.input_refs
        _assert_comparison_count(stack.session_factory, session_b.id, expected=1)

        _assert_mcp_surfaces(stack.service, session_b.id)
    finally:
        _cleanup(stack)


def _build_live_stack() -> LiveStack:
    from alembic import command
    from alembic.config import Config

    from api.chat_handler import ChatHandler
    from config.settings import settings
    from graph import build_graph
    from graph_conversation import build_conversation_graph
    from graph_discovery import build_discovery_graph
    from services.embeddings import OpenAIEmbeddingProvider
    from services.paperintel_service import PaperIntelService
    from services.qdrant_store import QdrantChunkStore
    from services.retrieval_layer import PostgresQdrantRetrievalLayer
    from storage.db import make_engine, make_session_factory
    from storage.repositories import (
        PostgresAgentRunPersistence,
        PostgresPaperChunkRepository,
        PostgresPaperWorkspaceRepository,
        PostgresSearchCandidateRepository,
        PostgresSessionStore,
        clear_foundation_tables,
    )

    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]
    qdrant_url = os.environ["PAPERINTEL_QDRANT_TEST_URL"]
    collection = f"paper_chunks_ca_live_{run_id}"
    print(f"LIVE_CA_RUN_ID={run_id}", flush=True)
    print(f"LIVE_CA_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_CA_PAPER_IDS={','.join(PAPER_IDS)}", flush=True)

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    vector_store = QdrantChunkStore.from_url(
        url=qdrant_url,
        collection_name=collection,
        vector_size=settings.openai_embedding_dimensions,
        timeout=30.0,
    )
    with session_factory() as db:
        clear_foundation_tables(db)

    artifact_repository = PostgresPaperWorkspaceRepository(session_factory)
    retrieval_layer = PostgresQdrantRetrievalLayer(
        chunk_repository=PostgresPaperChunkRepository(session_factory),
        vector_store=vector_store,
        embedding_provider=OpenAIEmbeddingProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            timeout=settings.openai_embedding_timeout,
        ),
    )
    session_store = PostgresSessionStore(session_factory)
    handler = ChatHandler(
        store=session_store,
        conversation_runner=build_conversation_graph(),
        analysis_runner=build_graph().compile(),
        discovery_runner=build_discovery_graph(),
        agent_run_persistence=PostgresAgentRunPersistence(session_factory),
        retrieval_layer=retrieval_layer,
        artifact_repository=artifact_repository,
    )
    service = PaperIntelService(
        handler=handler,
        candidate_repository=PostgresSearchCandidateRepository(session_factory),
        artifact_repository=artifact_repository,
    )
    return LiveStack(
        service=service,
        session_factory=session_factory,
        vector_store=vector_store,
        engine=engine,
        collection=collection,
    )


def _create_rest_app(service):
    from api.rest.app import create_rest_app

    return create_rest_app(service=service)


def _post_json(app, method: str, path: str, json: dict | None = None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(run())


def _require_local_pdfs() -> None:
    missing = [paper_id for paper_id in PAPER_IDS if not _pdf_path(paper_id).exists()]
    if missing:
        pytest.skip(
            "Local PDFs are required for CA live smoke: "
            + ",".join(str(_pdf_path(paper_id)) for paper_id in missing)
        )


def _pdf_path(paper_id: str) -> Path:
    return PDF_DIR / f"{paper_id}.pdf"


def _analyze_local_pair(service, session_id: str, *, label: str) -> None:
    for paper_id in PAPER_IDS:
        started = time.monotonic()
        result = service.analyze_pdf(
            session_id,
            str(_pdf_path(paper_id)),
            paper_id=paper_id,
            skip_arxiv_metadata_fetch=True,
        )
        elapsed = time.monotonic() - started
        print(
            f"LIVE_CA_SESSION_{label}_ANALYZE_{paper_id}_SECONDS={elapsed:.1f}",
            flush=True,
        )
        print(
            f"LIVE_CA_SESSION_{label}_ANALYZE_{paper_id}_PHASE={result.phase}",
            flush=True,
        )
        assert result.intent == "analyze_paper"
        assert result.phase == "qa", result.response_text


def _assert_workspace_count(service, session_id: str, *, expected: int, label: str) -> None:
    workspaces = service.list_paper_workspaces(session_id)
    workspace_ids = [workspace.paper_id for workspace in workspaces]
    print(
        f"LIVE_CA_SESSION_{label}_WORKSPACE_IDS={','.join(workspace_ids)}",
        flush=True,
    )
    assert len(workspaces) == expected
    assert workspace_ids == PAPER_IDS


def _latest_comparison_or_none(service, session_id: str):
    from services.paperintel_service import ComparisonNotFoundError

    try:
        return service.get_latest_comparison(session_id)
    except ComparisonNotFoundError:
        return None


def _assert_synthesis_response(payload: dict) -> None:
    selected = set(PAPER_IDS)
    assert payload["intent"] == "synthesis"
    assert payload["response_text"].strip()
    assert set(payload["referenced_paper_ids"]) == selected
    citation_ids = {citation["paper_id"] for citation in payload["citations"]}
    assert citation_ids
    assert citation_ids == selected


def _list_agent_runs(session_factory, session_id: str):
    from sqlalchemy import select

    from storage.mappers import orm_to_agent_run
    from storage.models import AgentRunORM

    with session_factory() as db:
        rows = (
            db.execute(
                select(AgentRunORM)
                .where(AgentRunORM.session_id == session_id)
                .order_by(AgentRunORM.started_at.asc())
            )
            .scalars()
            .all()
        )
    return [orm_to_agent_run(row) for row in rows]


def _require_run(runs, agent_name: str):
    matches = [run for run in runs if run.agent_name == agent_name]
    assert matches, f"Missing AgentRun for {agent_name}; got {[run.agent_name for run in runs]}"
    return matches[-1]


def _assert_completed_run(run, *, output_ref: str) -> None:
    print(
        f"LIVE_CA_AGENT_RUN={run.agent_name}:{run.id}:{run.status}:{run.output_ref}",
        flush=True,
    )
    if run.status == "fallback_used":
        pytest.fail(f"{run.agent_name} used fallback: {run.details}")
    assert run.status == "completed"
    assert run.output_ref == output_ref
    assert "policy_applied" in run.details


def _print_runs(label: str, runs) -> None:
    for run in runs:
        print(
            f"LIVE_CA_SESSION_{label}_RUN={run.agent_name}:{run.id}:{run.status}",
            flush=True,
        )


def _assert_comparison_count(session_factory, session_id: str, *, expected: int) -> None:
    from sqlalchemy import func, select

    from storage.models import ComparisonArtifactORM

    with session_factory() as db:
        count = db.execute(
            select(func.count())
            .select_from(ComparisonArtifactORM)
            .where(ComparisonArtifactORM.session_id == session_id)
        ).scalar_one()
    print(f"LIVE_CA_COMPARISON_COUNT_{session_id}={count}", flush=True)
    assert count == expected


def _assert_mcp_surfaces(service, session_id: str) -> None:
    from mcp_server.tools import compare_papers_tool, synthesize_papers_tool

    compare_output = asyncio.run(
        compare_papers_tool(
            service,
            session_id=session_id,
            paper_ids=PAPER_IDS,
            prompt="Check MCP comparison surface.",
        )
    )
    synthesis_output = asyncio.run(
        synthesize_papers_tool(
            service,
            session_id=session_id,
            prompt="Check MCP synthesis surface.",
        )
    )
    print(f"LIVE_CA_MCP_COMPARE_CHARS={len(compare_output)}", flush=True)
    print(f"LIVE_CA_MCP_SYNTHESIS_CHARS={len(synthesis_output)}", flush=True)
    assert "Latest persisted comparison" in compare_output
    assert "Paper Comparison" in compare_output or "Comparison" in compare_output
    assert "Synthesis for" in synthesis_output
    for marker in (
        "could not compare",
        "could not synthesize",
        "internal details",
        "Please try again.",
    ):
        assert marker not in compare_output
        assert marker not in synthesis_output


def _cleanup(stack: LiveStack) -> None:
    from storage.repositories import clear_foundation_tables

    try:
        stack.vector_store.client.delete_collection(collection_name=stack.collection)
        print("LIVE_CA_QDRANT_CLEANUP=success", flush=True)
    except Exception as exc:
        print(f"LIVE_CA_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

    try:
        with stack.session_factory() as db:
            clear_foundation_tables(db)
        print("LIVE_CA_POSTGRES_CLEANUP=success", flush=True)
    finally:
        stack.engine.dispose()
