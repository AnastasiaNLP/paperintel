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
    "PAPERINTEL_RUN_LIVE_PDF_SMOKE",
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
PAPER_ID = "1706.03762"
PDF_DIR = Path("~/Desktop/pdfs").expanduser()
REST_PAPER_ID = "local-rest-1706"
MCP_PAPER_ID = "local-mcp-1706"


def _missing_env() -> list[str]:
    missing = []
    for name in REQUIRED_ENV:
        value = os.environ.get(name)
        if name == "PAPERINTEL_RUN_LIVE_PDF_SMOKE":
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
            "PAPERINTEL_RUN_LIVE_PDF_SMOKE=1, PAPERINTEL_TEST_DATABASE_URL, "
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


def test_pdf_live_smoke_rest_upload_mcp_path_and_persistence():
    pdf_path = _require_local_pdf()
    if os.environ.get("PAPERINTEL_LIVE_TRACE") != "1":
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_PDF_SMOKE %(levelname)s %(name)s %(message)s",
        force=True,
    )

    stack = _build_live_stack()
    app = _create_rest_app(stack.service)

    try:
        rest_session = stack.service.create_session(persona="engineer")
        print(f"LIVE_PDF_REST_SESSION_ID={rest_session.id}", flush=True)

        started = time.monotonic()
        rest_response = _post_pdf_upload(
            app,
            f"/sessions/{rest_session.id}/analyze-pdf",
            pdf_path=pdf_path,
            paper_id=REST_PAPER_ID,
        )
        elapsed = time.monotonic() - started
        rest_payload = rest_response.json()
        print(f"LIVE_PDF_REST_ANALYZE_SECONDS={elapsed:.1f}", flush=True)
        print(f"LIVE_PDF_REST_STATUS={rest_response.status_code}", flush=True)
        assert rest_response.status_code == 200, rest_payload
        assert rest_payload["intent"] == "analyze_paper"
        assert rest_payload["phase"] == "qa"
        print(
            "LIVE_PDF_REST_REFERENCED="
            + ",".join(rest_payload.get("referenced_paper_ids") or []),
            flush=True,
        )
        _assert_workspace(stack.service, rest_session.id, REST_PAPER_ID, label="REST")

        mcp_session = stack.service.create_session(persona="researcher")
        print(f"LIVE_PDF_MCP_SESSION_ID={mcp_session.id}", flush=True)

        started = time.monotonic()
        mcp_output = _run_mcp_pdf_tool(stack.service, mcp_session.id, pdf_path)
        elapsed = time.monotonic() - started
        print(f"LIVE_PDF_MCP_ANALYZE_SECONDS={elapsed:.1f}", flush=True)
        print(f"LIVE_PDF_MCP_OUTPUT_CHARS={len(mcp_output)}", flush=True)
        assert "Paper analysis completed." in mcp_output
        for marker in ("could not", "internal details", "Please try again."):
            assert marker not in mcp_output
        _assert_workspace(stack.service, mcp_session.id, MCP_PAPER_ID, label="MCP")
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
    collection = f"paper_chunks_pdf_live_{run_id}"
    print(f"LIVE_PDF_RUN_ID={run_id}", flush=True)
    print(f"LIVE_PDF_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_PDF_SOURCE={_pdf_path()}", flush=True)

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


def _post_pdf_upload(app, path: str, *, pdf_path: Path, paper_id: str):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=180.0,
        ) as client:
            data = {
                "paper_id": paper_id,
                "skip_arxiv_metadata_fetch": "true",
            }
            with pdf_path.open("rb") as handle:
                files = {
                    "file": (pdf_path.name, handle, "application/pdf"),
                }
                return await client.post(path, data=data, files=files)

    return asyncio.run(run())


def _run_mcp_pdf_tool(service, session_id: str, pdf_path: Path) -> str:
    from mcp_server.tools import analyze_pdf_tool

    return asyncio.run(
        analyze_pdf_tool(
            service,
            session_id=session_id,
            pdf_path=str(pdf_path),
            paper_id=MCP_PAPER_ID,
            skip_arxiv_metadata_fetch=True,
        )
    )


def _assert_workspace(service, session_id: str, paper_id: str, *, label: str) -> None:
    workspaces = service.list_paper_workspaces(session_id)
    workspace_ids = [workspace.paper_id for workspace in workspaces]
    print(f"LIVE_PDF_{label}_WORKSPACE_IDS={','.join(workspace_ids)}", flush=True)
    assert workspace_ids == [paper_id]
    workspace = workspaces[0]
    print(f"LIVE_PDF_{label}_WORKSPACE_STAGE={workspace.pipeline_stage}", flush=True)
    assert workspace.pipeline_stage not in {"failed", "paper_failure_finalize"}
    assert workspace.full_markdown_report
    assert workspace.method_extraction_json is not None


def _require_local_pdf() -> Path:
    pdf_path = _pdf_path()
    if not pdf_path.exists():
        pytest.skip(f"Local PDF is required for PDF live smoke: {pdf_path}")
    return pdf_path


def _pdf_path() -> Path:
    return PDF_DIR / f"{PAPER_ID}.pdf"


def _cleanup(stack: LiveStack) -> None:
    from storage.repositories import clear_foundation_tables

    try:
        stack.vector_store.client.delete_collection(collection_name=stack.collection)
        print("LIVE_PDF_QDRANT_CLEANUP=success", flush=True)
    except Exception as exc:
        print(f"LIVE_PDF_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

    try:
        with stack.session_factory() as db:
            clear_foundation_tables(db)
        print("LIVE_PDF_POSTGRES_CLEANUP=success", flush=True)
    finally:
        stack.engine.dispose()
