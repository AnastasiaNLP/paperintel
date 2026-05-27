import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass

import httpx
import pytest


REQUIRED_ENV = (
    "PAPERINTEL_RUN_ASYNC_JOBS_LIVE",
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
PAPER_ID = "1706.03762"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ID}"


def _missing_env() -> list[str]:
    missing = []
    for name in REQUIRED_ENV:
        value = os.environ.get(name)
        if name == "PAPERINTEL_RUN_ASYNC_JOBS_LIVE":
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
            "PAPERINTEL_RUN_ASYNC_JOBS_LIVE=1, PAPERINTEL_TEST_DATABASE_URL, "
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
    job_repository: object


def test_async_jobs_live_smoke_rest_worker_mcp_and_persistence():
    if os.environ.get("PAPERINTEL_LIVE_TRACE") != "1":
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_AJ_SMOKE %(levelname)s %(name)s %(message)s",
        force=True,
    )

    stack = _build_live_stack()
    app = _create_rest_app(stack.service)

    try:
        session = stack.service.create_session(persona="engineer")
        print(f"LIVE_AJ_SESSION_ID={session.id}", flush=True)

        enqueue_response = _request_json(
            app,
            "POST",
            f"/sessions/{session.id}/jobs/analyze-paper",
            json={"paper_url": PAPER_URL},
        )
        enqueue_payload = enqueue_response.json()
        assert enqueue_response.status_code == 202, enqueue_payload
        assert enqueue_payload["status"] == "queued"
        assert enqueue_payload["kind"] == "analyze_paper"
        job_id = enqueue_payload["id"]
        print(f"LIVE_AJ_JOB_ID={job_id}", flush=True)

        started = time.monotonic()
        processed = _run_worker_once(stack, worker_id="live-aj-worker")
        elapsed = time.monotonic() - started
        assert processed is not None
        print(f"LIVE_AJ_WORKER_PROCESSED={processed.id}:{processed.status}", flush=True)
        print(f"LIVE_AJ_WORKER_SECONDS={elapsed:.1f}", flush=True)
        assert processed.id == job_id
        assert processed.status == "succeeded", processed.error_json

        status_response = _request_json(app, "GET", f"/jobs/{job_id}")
        status_payload = status_response.json()
        assert status_response.status_code == 200, status_payload
        print(f"LIVE_AJ_JOB_STATUS={status_payload['status']}", flush=True)
        assert status_payload["status"] == "succeeded"
        assert status_payload["result_json"]["session_id"] == session.id
        assert status_payload["result_json"]["phase"] == "qa"
        assert status_payload["result_json"]["intent"] == "analyze_paper"

        workspaces = stack.service.list_paper_workspaces(session.id)
        workspace_ids = [workspace.paper_id for workspace in workspaces]
        print(f"LIVE_AJ_WORKSPACE_IDS={','.join(workspace_ids)}", flush=True)
        assert PAPER_ID in workspace_ids

        _assert_mcp_job_status(stack.service, session.id, job_id)
        _assert_invalid_input_failure(app, stack, session.id)
        _assert_cancel_path(app, stack, session.id)
    finally:
        _cleanup(stack)


def _build_live_stack() -> LiveStack:
    from alembic import command
    from alembic.config import Config

    from api.chat_handler import ChatHandler
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
        PostgresWorkflowJobRepository,
        clear_foundation_tables,
    )

    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]
    qdrant_url = os.environ["PAPERINTEL_QDRANT_TEST_URL"]
    collection = f"paper_chunks_aj_live_{run_id}"
    print(f"LIVE_AJ_RUN_ID={run_id}", flush=True)
    print(f"LIVE_AJ_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_AJ_PAPER_URL={PAPER_URL}", flush=True)

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
    with session_factory() as db:
        clear_foundation_tables(db)

    artifact_repository = PostgresPaperWorkspaceRepository(session_factory)
    job_repository = PostgresWorkflowJobRepository(session_factory)
    retrieval_layer = PostgresQdrantRetrievalLayer(
        chunk_repository=PostgresPaperChunkRepository(session_factory),
        vector_store=vector_store,
        embedding_provider=OpenAIEmbeddingProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=60.0,
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
        workflow_job_repository=job_repository,
    )
    return LiveStack(
        service=service,
        session_factory=session_factory,
        vector_store=vector_store,
        engine=engine,
        collection=collection,
        job_repository=job_repository,
    )


def _create_rest_app(service):
    from api.rest.app import create_rest_app

    return create_rest_app(service=service)


def _request_json(app, method: str, path: str, json: dict | None = None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(run())


def _run_worker_once(stack: LiveStack, *, worker_id: str):
    from workers.workflow_worker import WorkflowJobExecutor, WorkflowWorker

    worker = WorkflowWorker(
        repository=stack.job_repository,
        executor=WorkflowJobExecutor(stack.service),
        worker_id=worker_id,
    )
    return worker.run_once()


def _assert_mcp_job_status(service, session_id: str, job_id: str) -> None:
    from mcp_server.tools import get_workflow_job_tool, list_workflow_jobs_tool

    job_output = asyncio.run(get_workflow_job_tool(service, job_id=job_id))
    list_output = asyncio.run(list_workflow_jobs_tool(service, session_id=session_id))
    print(f"LIVE_AJ_MCP_JOB_STATUS_CHARS={len(job_output)}", flush=True)
    print(f"LIVE_AJ_MCP_JOB_LIST_CHARS={len(list_output)}", flush=True)
    assert "Workflow job" in job_output
    assert "Kind: analyze_paper" in job_output
    assert "Status: succeeded" in job_output
    assert job_id in list_output
    for marker in ("could not", "internal details", "Please try again."):
        assert marker not in job_output
        assert marker not in list_output


def _assert_invalid_input_failure(app, stack: LiveStack, session_id: str) -> None:
    from models.jobs import WorkflowJob

    bad_job = stack.job_repository.create(
        WorkflowJob(
            session_id=session_id,
            kind="analyze_paper",
            input_json={},
        )
    )
    processed = _run_worker_once(stack, worker_id="live-aj-worker-invalid")
    assert processed is not None
    print(f"LIVE_AJ_INVALID_JOB={processed.id}:{processed.status}", flush=True)
    assert processed.id == bad_job.id
    assert processed.status == "failed"
    assert processed.error_json["error"] == "invalid_job_input"

    response = _request_json(app, "GET", f"/jobs/{bad_job.id}")
    payload = response.json()
    assert response.status_code == 200, payload
    assert payload["status"] == "failed"
    assert payload["error_json"]["error"] == "invalid_job_input"


def _assert_cancel_path(app, stack: LiveStack, session_id: str) -> None:
    enqueue_response = _request_json(
        app,
        "POST",
        f"/sessions/{session_id}/jobs/analyze-paper",
        json={"paper_url": PAPER_URL},
    )
    assert enqueue_response.status_code == 202, enqueue_response.json()
    job_id = enqueue_response.json()["id"]

    cancel_response = _request_json(app, "POST", f"/jobs/{job_id}/cancel")
    cancel_payload = cancel_response.json()
    print(f"LIVE_AJ_CANCELED_JOB={job_id}:{cancel_payload['status']}", flush=True)
    assert cancel_response.status_code == 200, cancel_payload
    assert cancel_payload["status"] == "canceled"

    processed = _run_worker_once(stack, worker_id="live-aj-worker-cancel")
    print(f"LIVE_AJ_CANCELED_WORKER_RESULT={processed}", flush=True)
    assert processed is None


def _cleanup(stack: LiveStack) -> None:
    from storage.repositories import clear_foundation_tables

    try:
        stack.vector_store.client.delete_collection(collection_name=stack.collection)
        print("LIVE_AJ_QDRANT_CLEANUP=success", flush=True)
    except Exception as exc:
        print(f"LIVE_AJ_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

    try:
        with stack.session_factory() as db:
            clear_foundation_tables(db)
        print("LIVE_AJ_POSTGRES_CLEANUP=success", flush=True)
    finally:
        stack.engine.dispose()
