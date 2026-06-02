from datetime import datetime, timezone

from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_artifacts import BlobArtifact, BlobReference
from models.discovery import SearchCandidate
from models.external_metadata import ArxivMetadataCacheEntry
from models.errors import ErrorCodes, StructuredError, make_error
from models.jobs import WorkflowJob
from models.pdf_uploads import PdfUpload
from models.retrieval import ChunkLocation, ChunkSource, EvidenceArtifact, PaperChunk
from models.session import Session, Turn
from storage.mappers import (
    agent_run_to_orm,
    arxiv_metadata_cache_entry_to_orm,
    blob_artifact_to_orm,
    blob_reference_to_orm,
    comparison_artifact_to_orm,
    orm_to_agent_run,
    orm_to_arxiv_metadata_cache_entry,
    orm_to_blob_artifact,
    orm_to_blob_reference,
    orm_to_comparison_artifact,
    orm_to_paper_chunk,
    orm_to_paper_workspace,
    orm_to_pdf_upload,
    orm_to_session,
    orm_to_search_candidate,
    orm_to_structured_error,
    orm_to_turn,
    paper_chunk_to_orm,
    paper_workspace_to_orm,
    pdf_upload_to_orm,
    search_candidate_to_orm,
    session_to_orm,
    structured_error_to_orm,
    turn_to_orm,
    workflow_job_to_orm,
    orm_to_workflow_job,
)


def test_structured_error_has_stable_id_by_default():
    first = make_error(ErrorCodes.WARNING, "warning")
    second = make_error(ErrorCodes.WARNING, "warning")

    assert first.id
    assert second.id
    assert first.id != second.id


def test_session_mapper_round_trip():
    session = Session(
        persona="researcher",
        original_query="memory agents",
        phase="qa",
        selected_candidate_ids=["c1"],
        active_paper_ids=["p1"],
        latest_comparison_id="cmp-1",
    )

    mapped = orm_to_session(session_to_orm(session))

    assert mapped == session


def test_structured_error_mapper_round_trip():
    error = StructuredError(
        code=ErrorCodes.FATAL_ERROR,
        message="boom",
        node="handler",
        severity="fatal",
        recoverable=False,
        session_id="session-1",
        details={"exception_type": "RuntimeError"},
    )

    mapped = orm_to_structured_error(structured_error_to_orm(error))

    assert mapped == error


def test_turn_mapper_round_trip_with_error():
    error = make_error(ErrorCodes.WARNING, "warning", session_id="session-1")
    turn = Turn(
        session_id="session-1",
        role="assistant",
        content="response",
        intent="qa",
        referenced_paper_ids=["paper-1"],
        artifact_refs=["artifact-1"],
        error=error,
        metadata={"source": "test"},
    )
    orm = turn_to_orm(turn, error_id=error.id)
    orm.error = structured_error_to_orm(error)

    mapped = orm_to_turn(orm)

    assert mapped == turn


def test_agent_run_mapper_round_trip():
    run = AgentRun(
        session_id="session-1",
        job_id="job-1",
        agent_name="report",
        input_refs=["state:report"],
        model="claude-haiku",
        tool_calls=[{"tool": "search", "args": {"q": "x"}}],
        iteration_count=1,
        llm_call_count=2,
        details={"policy_warning": "exceeded_max_tool_calls"},
    )
    run.complete(output_ref="state:report", tokens_used=100, cost_usd=0.01)

    mapped = orm_to_agent_run(agent_run_to_orm(run))

    assert mapped == run


def test_paper_chunk_mapper_round_trip():
    chunk = PaperChunk(
        id="2310.06825:chunk:0",
        paper_id="2310.06825",
        chunk_index=0,
        text="Table 1 reports retrieval quality.",
        chunk_type="table",
        source=ChunkSource(
            paper_id="2310.06825",
            session_id="session-1",
            paper_index=0,
            input_url="https://arxiv.org/abs/2310.06825",
            arxiv_id="2310.06825",
        ),
        location=ChunkLocation(page_start=4, page_end=4, section_title="Results"),
        artifact_refs=[
            EvidenceArtifact(
                paper_id="2310.06825",
                artifact_type="table",
                storage_ref="s3://paperintel/table-1.png",
            )
        ],
        metadata={"header_context": "Results"},
    )

    mapped = orm_to_paper_chunk(paper_chunk_to_orm(chunk))

    assert mapped == chunk


def test_search_candidate_mapper_round_trip():
    candidate = SearchCandidate(
        session_id="session-1",
        discovery_turn_id="turn-1",
        display_rank=1,
        status="selected",
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        authors=["Ashish Vaswani"],
        year=2017,
        arxiv_id="1706.03762",
        abstract="Transformer paper.",
        published_at=datetime(2017, 6, 12, tzinfo=timezone.utc),
        score=0.95,
        reasons=["exact title match"],
        metadata={"provider_rank": 1},
    )

    mapped = orm_to_search_candidate(search_candidate_to_orm(candidate))

    assert mapped == candidate


def test_paper_workspace_mapper_round_trip():
    workspace = PaperWorkspace(
        session_id="session-1",
        paper_id="1706.03762",
        title="Attention Is All You Need",
        source_url="https://arxiv.org/abs/1706.03762",
        pipeline_stage="chunk_and_index",
        pipeline_version="pipeline-v2",
        finalized_report_json={"recommended_action": "prototype"},
        method_extraction_json={"method_name": "Transformer"},
        benchmarks_json=[{"task": "translation", "metric": "BLEU"}],
        readiness_json={"maturity_level": "experimental"},
        full_markdown_report="# Report",
    )

    mapped = orm_to_paper_workspace(paper_workspace_to_orm(workspace))

    assert mapped == workspace


def test_comparison_artifact_mapper_round_trip():
    artifact = ComparisonArtifact(
        session_id="session-1",
        paper_ids=["1706.03762", "2605.16113"],
        comparison_report_json={"winner_basis": "readiness_dominant"},
        comparison_markdown="# Paper Comparison",
    )

    mapped = orm_to_comparison_artifact(comparison_artifact_to_orm(artifact))

    assert mapped == artifact


def test_workflow_job_mapper_round_trip():
    job = WorkflowJob(
        session_id="session-1",
        kind="analyze_paper",
        status="running",
        input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
        result_json={"phase": "qa"},
        attempts=1,
        max_attempts=2,
        idempotency_key="session-1:blob-1:v1",
        pipeline_version="v1",
        next_attempt_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        retry_policy_json={"base_delay_seconds": 10},
        locked_by="worker-1",
        lease_expires_at=datetime(2026, 6, 2, 0, 5, tzinfo=timezone.utc),
        heartbeat_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    mapped = orm_to_workflow_job(workflow_job_to_orm(job))

    assert mapped == job


def test_arxiv_metadata_cache_mapper_round_trip():
    fetched_at = datetime(2017, 6, 12, tzinfo=timezone.utc)
    entry = ArxivMetadataCacheEntry(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="Transformer paper.",
        published_date="2017-06-12T17:57:34Z",
        categories=["cs.CL", "cs.LG"],
        source_url="https://arxiv.org/abs/1706.03762",
        fetched_at=fetched_at,
        last_error_json={"code": "429", "message": "rate limited"},
        error_count=2,
    )

    mapped = orm_to_arxiv_metadata_cache_entry(
        arxiv_metadata_cache_entry_to_orm(entry)
    )

    assert mapped == entry
    assert mapped.has_successful_fetch is True


def test_blob_artifact_mapper_round_trip():
    artifact = BlobArtifact(
        kind="pdf",
        object_key="papers/sha256/aa/" + "a" * 64 + ".pdf",
        bucket_name="paperintel",
        content_hash="a" * 64,
        content_type="application/pdf",
        size_bytes=128,
    )

    mapped = orm_to_blob_artifact(blob_artifact_to_orm(artifact))

    assert mapped == artifact


def test_blob_reference_mapper_round_trip():
    reference = BlobReference(
        blob_id="blob-1",
        ref_kind="session",
        ref_id="session-1",
        metadata={"source": "upload"},
    )

    mapped = orm_to_blob_reference(blob_reference_to_orm(reference))

    assert mapped == reference


def test_pdf_upload_mapper_round_trip():
    finalized_at = datetime(2026, 6, 2, tzinfo=timezone.utc)
    upload = PdfUpload(
        session_id="session-1",
        blob_id="blob-1",
        object_key="uploads/session-1/upload-1.pdf",
        expected_sha256="a" * 64,
        actual_sha256="a" * 64,
        size_bytes=128,
        status="finalized",
        expires_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
        finalized_at=finalized_at,
    )

    mapped = orm_to_pdf_upload(pdf_upload_to_orm(upload))

    assert mapped == upload
