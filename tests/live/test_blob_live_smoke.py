import asyncio
import hashlib
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select


REQUIRED_ENV = (
    "PAPERINTEL_RUN_LIVE_BLOB_SMOKE",
    "PAPERINTEL_TEST_DATABASE_URL",
    "PAPERINTEL_QDRANT_TEST_URL",
    "PAPERINTEL_MINIO_TEST_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
PAPER_ID = "1706.03762"
PDF_DIR = Path("~/Desktop/pdfs").expanduser()
REST_PAPER_ID = "local-rest-blob-1706"
MCP_PAPER_ID = "local-mcp-blob-1706"
ASYNC_PAPER_ID = "local-async-blob-1706"


def _missing_env() -> list[str]:
    missing = []
    for name in REQUIRED_ENV:
        value = os.environ.get(name)
        if name == "PAPERINTEL_RUN_LIVE_BLOB_SMOKE":
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
            "PAPERINTEL_RUN_LIVE_BLOB_SMOKE=1, PAPERINTEL_TEST_DATABASE_URL, "
            "PAPERINTEL_QDRANT_TEST_URL, PAPERINTEL_MINIO_TEST_URL, "
            "ANTHROPIC_API_KEY, and OPENAI_API_KEY are required"
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
    bucket: str


def test_blob_live_smoke_factory_rest_mcp_dedup_health_and_cleanup(monkeypatch):
    pdf_path = _require_local_pdf()
    if os.environ.get("PAPERINTEL_LIVE_TRACE") != "1":
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
    logging.basicConfig(
        level=logging.INFO,
        format="LIVE_BLOB_SMOKE %(levelname)s %(name)s %(message)s",
        force=True,
    )

    stack = _build_live_stack(monkeypatch)
    app = _create_rest_app(stack.service)
    initial_temp_paths = _blob_temp_paths()

    try:
        rest_session = stack.service.create_session(persona="engineer")
        print(f"LIVE_BLOB_REST_SESSION_ID={rest_session.id}", flush=True)
        started = time.monotonic()
        rest_response = _post_pdf_upload(
            app,
            f"/sessions/{rest_session.id}/analyze-pdf",
            pdf_path=pdf_path,
            paper_id=REST_PAPER_ID,
        )
        print(
            f"LIVE_BLOB_REST_ANALYZE_SECONDS={time.monotonic() - started:.1f}",
            flush=True,
        )
        print(f"LIVE_BLOB_REST_STATUS={rest_response.status_code}", flush=True)
        assert rest_response.status_code == 200, rest_response.json()
        _assert_workspace(stack.service, rest_session.id, REST_PAPER_ID)
        _assert_no_new_temp_paths(initial_temp_paths)

        mcp_session = stack.service.create_session(persona="researcher")
        print(f"LIVE_BLOB_MCP_SESSION_ID={mcp_session.id}", flush=True)
        started = time.monotonic()
        mcp_output = _run_mcp_pdf_tool(stack.service, mcp_session.id, pdf_path)
        print(
            f"LIVE_BLOB_MCP_ANALYZE_SECONDS={time.monotonic() - started:.1f}",
            flush=True,
        )
        print(f"LIVE_BLOB_MCP_OUTPUT_CHARS={len(mcp_output)}", flush=True)
        assert "Paper analysis completed." in mcp_output
        for marker in ("could not", "internal details", "Please try again."):
            assert marker not in mcp_output
        _assert_workspace(stack.service, mcp_session.id, MCP_PAPER_ID)
        _assert_no_new_temp_paths(initial_temp_paths)

        async_session = stack.service.create_session(persona="engineer")
        print(f"LIVE_BLOB_ASYNC_SESSION_ID={async_session.id}", flush=True)
        async_job_id = _run_async_pdf_upload_flow(
            app,
            stack,
            async_session.id,
            pdf_path=pdf_path,
            paper_id=ASYNC_PAPER_ID,
        )
        _assert_workspace(stack.service, async_session.id, ASYNC_PAPER_ID)
        _assert_no_new_temp_paths(initial_temp_paths)

        canceled_session = stack.service.create_session(persona="engineer")
        print(f"LIVE_BLOB_CANCEL_SESSION_ID={canceled_session.id}", flush=True)
        _assert_async_pdf_cancel_flow(
            app,
            stack,
            canceled_session.id,
            pdf_path=pdf_path,
        )

        artifact = _assert_blob_registry(
            stack,
            pdf_path=pdf_path,
            expected_session_ids={
                rest_session.id,
                mcp_session.id,
                async_session.id,
                canceled_session.id,
            },
            expected_reference_count=9,
        )
        print(f"LIVE_BLOB_ARTIFACT_ID={artifact.id}", flush=True)
        print(f"LIVE_BLOB_OBJECT_KEY={artifact.object_key}", flush=True)
        print("LIVE_BLOB_ARTIFACT_COUNT=1", flush=True)
        print("LIVE_BLOB_REFERENCE_COUNT=9", flush=True)
        print("LIVE_BLOB_OBJECT_COUNT=1", flush=True)
        print(f"LIVE_BLOB_ASYNC_JOB_ID={async_job_id}", flush=True)

        health_response = _get(app, "/health")
        health_payload = health_response.json()
        print(f"LIVE_BLOB_HEALTH_STATUS={health_response.status_code}", flush=True)
        print(
            f"LIVE_BLOB_HEALTH_STORE={health_payload['checks']['blob_store']}",
            flush=True,
        )
        assert health_response.status_code == 200, health_payload
        assert health_payload["status"] == "healthy"
        assert health_payload["checks"]["blob_store"] == "ok"
    finally:
        _cleanup(stack)


def _build_live_stack(monkeypatch) -> LiveStack:
    run_id = uuid.uuid4().hex[:12]
    database_url = os.environ["PAPERINTEL_TEST_DATABASE_URL"]
    qdrant_url = os.environ["PAPERINTEL_QDRANT_TEST_URL"]
    minio_url = os.environ["PAPERINTEL_MINIO_TEST_URL"]
    collection = f"paper_chunks_blob_live_{run_id}"
    bucket = f"paperintel-blob-live-{run_id}"
    print(f"LIVE_BLOB_RUN_ID={run_id}", flush=True)
    print(f"LIVE_BLOB_QDRANT_COLLECTION={collection}", flush=True)
    print(f"LIVE_BLOB_MINIO_BUCKET={bucket}", flush=True)
    print(f"LIVE_BLOB_SOURCE={_pdf_path()}", flush=True)

    monkeypatch.setenv("BLOB_STORAGE_ENABLED", "true")
    monkeypatch.setenv("BLOB_S3_ENDPOINT_URL", minio_url)
    monkeypatch.setenv("BLOB_S3_REGION", "us-east-1")
    monkeypatch.setenv("BLOB_S3_BUCKET", bucket)
    monkeypatch.setenv(
        "BLOB_S3_ACCESS_KEY_ID",
        os.environ.get("PAPERINTEL_MINIO_TEST_ACCESS_KEY_ID", "paperintel"),
    )
    monkeypatch.setenv(
        "BLOB_S3_SECRET_ACCESS_KEY",
        os.environ.get(
            "PAPERINTEL_MINIO_TEST_SECRET_ACCESS_KEY",
            "paperintel_dev_password",
        ),
    )

    from alembic import command
    from alembic.config import Config

    import config.settings as settings_module
    from api.app_factory import create_paperintel_service
    from config.settings import Settings
    from services.qdrant_store import QdrantChunkStore
    from storage.db import make_engine, make_session_factory
    from storage.repositories import clear_foundation_tables

    monkeypatch.setattr(settings_module, "settings", Settings())
    engine = None
    session_factory = None
    vector_store = None
    try:
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(config, "head")

        engine = make_engine(database_url)
        session_factory = make_session_factory(engine)
        with session_factory() as db:
            clear_foundation_tables(db)

        service = create_paperintel_service(
            database_url=database_url,
            qdrant_url=qdrant_url,
            qdrant_collection=collection,
        )
        assert service.blob_store.bucket_name == bucket
        vector_store = QdrantChunkStore.from_url(
            url=qdrant_url,
            collection_name=collection,
            timeout=30.0,
        )
        return LiveStack(
            service=service,
            session_factory=session_factory,
            vector_store=vector_store,
            engine=engine,
            collection=collection,
            bucket=bucket,
        )
    except Exception:
        _cleanup_partial_stack(
            collection=collection,
            bucket=bucket,
            vector_store=vector_store,
            session_factory=session_factory,
            engine=engine,
        )
        raise


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
                files = {"file": (pdf_path.name, handle, "application/pdf")}
                return await client.post(path, data=data, files=files)

    return asyncio.run(run())


def _get(app, path: str):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    return asyncio.run(run())


def _post_json(app, path: str, *, payload: dict | None = None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(path, json=payload)

    return asyncio.run(run())


def _put_presigned(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
    return httpx.put(url, content=content, headers=headers, timeout=60.0)


def _run_worker_once(stack: LiveStack, *, worker_id: str):
    from workers.workflow_worker import WorkflowJobExecutor, WorkflowWorker

    return WorkflowWorker(
        repository=stack.service.workflow_job_repository,
        executor=WorkflowJobExecutor(stack.service),
        worker_id=worker_id,
        kinds=["analyze_pdf_blob"],
    ).run_once()


def _run_async_pdf_upload_flow(
    app,
    stack: LiveStack,
    session_id: str,
    *,
    pdf_path: Path,
    paper_id: str,
) -> str:
    content = pdf_path.read_bytes()
    initiate_response = _post_json(
        app,
        f"/sessions/{session_id}/pdf-uploads",
        payload={
            "expected_sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        },
    )
    initiate_payload = initiate_response.json()
    print(f"LIVE_BLOB_ASYNC_INITIATE_STATUS={initiate_response.status_code}", flush=True)
    assert initiate_response.status_code == 201, initiate_payload
    upload = initiate_payload["upload"]

    put_response = _put_presigned(
        initiate_payload["upload_url"],
        content=content,
        headers=initiate_payload["upload_headers"],
    )
    print(f"LIVE_BLOB_ASYNC_PUT_STATUS={put_response.status_code}", flush=True)
    assert put_response.status_code in {200, 204}, put_response.text

    finalize_response = _post_json(
        app,
        f"/sessions/{session_id}/pdf-uploads/{upload['id']}/finalize",
    )
    finalize_payload = finalize_response.json()
    print(f"LIVE_BLOB_ASYNC_FINALIZE_STATUS={finalize_response.status_code}", flush=True)
    assert finalize_response.status_code == 200, finalize_payload
    assert finalize_payload["status"] == "finalized"
    assert finalize_payload["blob_id"]

    enqueue_response = _post_json(
        app,
        f"/sessions/{session_id}/pdf-uploads/{upload['id']}/jobs/analyze",
        payload={
            "paper_id": paper_id,
            "skip_arxiv_metadata_fetch": True,
            "pipeline_version": "live-async-pdf",
        },
    )
    enqueue_payload = enqueue_response.json()
    print(f"LIVE_BLOB_ASYNC_ENQUEUE_STATUS={enqueue_response.status_code}", flush=True)
    assert enqueue_response.status_code == 202, enqueue_payload
    assert enqueue_payload["kind"] == "analyze_pdf_blob"
    assert enqueue_payload["status"] == "queued"
    job_id = enqueue_payload["id"]

    started = time.monotonic()
    processed = _run_worker_once(stack, worker_id="live-blob-async-worker")
    print(f"LIVE_BLOB_ASYNC_WORKER_SECONDS={time.monotonic() - started:.1f}", flush=True)
    assert processed is not None
    print(f"LIVE_BLOB_ASYNC_WORKER_RESULT={processed.id}:{processed.status}", flush=True)
    assert processed.id == job_id
    assert processed.status == "succeeded"

    job_response = _get(app, f"/jobs/{job_id}")
    job_payload = job_response.json()
    print(f"LIVE_BLOB_ASYNC_JOB_STATUS={job_payload['status']}", flush=True)
    assert job_response.status_code == 200, job_payload
    assert job_payload["status"] == "succeeded"
    return job_id


def _assert_async_pdf_cancel_flow(
    app,
    stack: LiveStack,
    session_id: str,
    *,
    pdf_path: Path,
) -> None:
    content = pdf_path.read_bytes()
    upload = stack.service.store_pdf_upload(session_id, content)
    job = stack.service.enqueue_analyze_pdf_blob(
        session_id,
        upload.id,
        paper_id="local-canceled-blob-1706",
        skip_arxiv_metadata_fetch=True,
        pipeline_version="live-async-pdf-cancel",
    )
    cancel_response = _post_json(app, f"/jobs/{job.id}/cancel")
    cancel_payload = cancel_response.json()
    print(f"LIVE_BLOB_ASYNC_CANCEL_STATUS={cancel_payload['status']}", flush=True)
    assert cancel_response.status_code == 200, cancel_payload
    assert cancel_payload["status"] == "canceled"
    processed = _run_worker_once(stack, worker_id="live-blob-cancel-worker")
    print(f"LIVE_BLOB_ASYNC_CANCEL_WORKER_RESULT={processed}", flush=True)
    assert processed is None


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


def _assert_workspace(service, session_id: str, paper_id: str) -> None:
    workspace_ids = [
        workspace.paper_id for workspace in service.list_paper_workspaces(session_id)
    ]
    print(f"LIVE_BLOB_WORKSPACE_IDS_{session_id}={','.join(workspace_ids)}", flush=True)
    assert workspace_ids == [paper_id]


def _assert_blob_registry(
    stack: LiveStack,
    *,
    pdf_path: Path,
    expected_session_ids: set[str],
    expected_reference_count: int,
):
    from storage.models import BlobArtifactORM, BlobReferenceORM

    expected_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    repository = stack.service.blob_artifact_repository
    artifact = repository.get_by_kind_and_hash("pdf", expected_hash)
    assert artifact is not None
    references = repository.list_references(artifact.id)
    with stack.session_factory() as db:
        artifact_count = db.execute(select(func.count(BlobArtifactORM.id))).scalar_one()
        reference_count = db.execute(select(func.count(BlobReferenceORM.id))).scalar_one()
    assert artifact_count == 1
    assert reference_count == expected_reference_count
    assert [reference.ref_kind for reference in references].count("session") == 4
    assert [reference.ref_kind for reference in references].count("paper_workspace") == 3
    assert [reference.ref_kind for reference in references].count("workflow_job") == 2
    assert [
        reference.status
        for reference in references
        if reference.ref_kind == "workflow_job"
    ] == ["released", "released"]
    assert {
        reference.ref_id for reference in references if reference.ref_kind == "session"
    } == expected_session_ids
    response = stack.service.blob_store.client.list_objects_v2(Bucket=stack.bucket)
    assert len(response.get("Contents", [])) == 1
    return artifact


def _blob_temp_paths() -> set[Path]:
    temp_dir = Path(tempfile.gettempdir())
    return set(temp_dir.glob("paperintel_blob_*")) | set(
        temp_dir.glob("paperintel_upload_*")
    )


def _assert_no_new_temp_paths(initial_paths: set[Path]) -> None:
    assert _blob_temp_paths() == initial_paths


def _require_local_pdf() -> Path:
    pdf_path = _pdf_path()
    if not pdf_path.exists():
        pytest.skip(f"Local PDF is required for blob-storage live smoke: {pdf_path}")
    return pdf_path


def _pdf_path() -> Path:
    return PDF_DIR / f"{PAPER_ID}.pdf"


def _cleanup(stack: LiveStack) -> None:
    _cleanup_partial_stack(
        collection=stack.collection,
        bucket=stack.bucket,
        vector_store=stack.vector_store,
        session_factory=stack.session_factory,
        engine=stack.engine,
    )


def _cleanup_partial_stack(
    *,
    collection: str,
    bucket: str,
    vector_store=None,
    session_factory=None,
    engine=None,
) -> None:
    from storage.repositories import clear_foundation_tables

    try:
        if vector_store is not None:
            vector_store.client.delete_collection(collection_name=collection)
        print("LIVE_BLOB_QDRANT_CLEANUP=success", flush=True)
    except Exception as exc:
        print(f"LIVE_BLOB_QDRANT_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

    try:
        _delete_minio_bucket(bucket)
        print("LIVE_BLOB_MINIO_CLEANUP=success", flush=True)
    except Exception as exc:
        print(f"LIVE_BLOB_MINIO_CLEANUP=failed:{type(exc).__name__}:{exc}", flush=True)

    try:
        if session_factory is not None:
            with session_factory() as db:
                clear_foundation_tables(db)
        print("LIVE_BLOB_POSTGRES_CLEANUP=success", flush=True)
    finally:
        if engine is not None:
            engine.dispose()


def _delete_minio_bucket(bucket: str) -> None:
    client = _minio_client()
    try:
        response = client.list_objects_v2(Bucket=bucket)
    except client.exceptions.NoSuchBucket:
        return
    for item in response.get("Contents", []):
        client.delete_object(Bucket=bucket, Key=item["Key"])
    client.delete_bucket(Bucket=bucket)


def _minio_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["PAPERINTEL_MINIO_TEST_URL"],
        region_name="us-east-1",
        aws_access_key_id=os.environ.get(
            "PAPERINTEL_MINIO_TEST_ACCESS_KEY_ID",
            "paperintel",
        ),
        aws_secret_access_key=os.environ.get(
            "PAPERINTEL_MINIO_TEST_SECRET_ACCESS_KEY",
            "paperintel_dev_password",
        ),
    )
