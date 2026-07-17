from sqlalchemy.dialects import postgresql

from storage.models import (
    AgentRunORM,
    ArxivMetadataCacheORM,
    Base,
    BlobArtifactORM,
    BlobReferenceORM,
    ComparisonArtifactORM,
    PaperChunkORM,
    PaperWorkspaceORM,
    PdfUploadORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
    WorkflowJobORM,
)


def _postgres_type(column):
    return column.type.dialect_impl(postgresql.dialect())


def test_initial_storage_metadata_contains_foundation_tables():
    assert {
        "sessions",
        "turns",
        "agent_runs",
        "structured_errors",
        "paper_chunks",
        "search_candidates",
        "paper_workspaces",
        "comparison_artifacts",
        "workflow_jobs",
        "arxiv_metadata_cache",
        "blob_artifacts",
        "blob_references",
        "pdf_uploads",
    }.issubset(Base.metadata.tables.keys())


def test_session_table_has_phase_and_json_metadata_columns():
    table = SessionORM.__table__

    assert table.c.id.primary_key
    assert table.c.persona.nullable is False
    assert table.c.phase.nullable is False
    assert isinstance(_postgres_type(table.c.selected_candidate_ids), postgresql.JSONB)
    assert isinstance(_postgres_type(table.c.active_paper_ids), postgresql.JSONB)


def test_turn_table_links_to_session_and_structured_error():
    table = TurnORM.__table__

    foreign_keys = {fk.target_fullname for fk in table.foreign_keys}
    assert "sessions.id" in foreign_keys
    assert "structured_errors.id" in foreign_keys
    assert isinstance(_postgres_type(table.c.referenced_paper_ids), postgresql.JSONB)
    assert isinstance(_postgres_type(table.c.metadata_json), postgresql.JSONB)


def test_agent_run_table_matches_agent_run_contract_columns():
    columns = AgentRunORM.__table__.c

    for name in [
        "agent_name",
        "input_refs",
        "output_ref",
        "model",
        "tool_calls",
        "iteration_count",
        "llm_call_count",
        "termination_reason",
        "status",
        "details_json",
        "started_at",
        "finished_at",
    ]:
        assert name in columns

    assert isinstance(_postgres_type(columns.details_json), postgresql.JSONB)


def test_structured_error_table_matches_error_contract_columns():
    columns = StructuredErrorORM.__table__.c

    for name in [
        "code",
        "message",
        "node",
        "agent",
        "severity",
        "recoverable",
        "session_id",
        "paper_id",
        "agent_run_id",
        "details_json",
    ]:
        assert name in columns


def test_paper_chunk_table_matches_retrieval_contract_columns():
    columns = PaperChunkORM.__table__.c

    for name in [
        "paper_id",
        "session_id",
        "paper_index",
        "chunk_index",
        "chunk_type",
        "text",
        "source_json",
        "location_json",
        "artifact_refs_json",
        "metadata_json",
        "embedding_model",
        "embedding_dimensions",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert isinstance(_postgres_type(columns.source_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.location_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.artifact_refs_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.metadata_json), postgresql.JSONB)


def test_search_candidate_table_matches_discovery_contract_columns():
    columns = SearchCandidateORM.__table__.c

    for name in [
        "session_id",
        "discovery_turn_id",
        "display_rank",
        "status",
        "title",
        "url",
        "source",
        "authors",
        "year",
        "arxiv_id",
        "abstract",
        "published_at",
        "score",
        "reasons",
        "metadata_json",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert "ck_search_candidates_status" in {
        constraint.name for constraint in SearchCandidateORM.__table__.constraints
    }
    assert isinstance(_postgres_type(columns.authors), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.reasons), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.metadata_json), postgresql.JSONB)


def test_paper_workspace_table_matches_artifact_contract_columns():
    columns = PaperWorkspaceORM.__table__.c

    for name in [
        "session_id",
        "paper_id",
        "title",
        "source_url",
        "pipeline_stage",
        "pipeline_version",
        "finalized_report_json",
        "method_extraction_json",
        "benchmarks_json",
        "benchmark_candidates_json",
        "benchmark_extractor_version",
        "readiness_json",
        "full_markdown_report",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert "uq_paper_workspaces_session_paper" in {
        constraint.name for constraint in PaperWorkspaceORM.__table__.constraints
    }
    assert isinstance(_postgres_type(columns.finalized_report_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.method_extraction_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.benchmarks_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.benchmark_candidates_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.readiness_json), postgresql.JSONB)


def test_comparison_artifact_table_matches_group_artifact_contract_columns():
    columns = ComparisonArtifactORM.__table__.c

    for name in [
        "session_id",
        "paper_ids",
        "comparison_report_json",
        "comparison_markdown",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert isinstance(_postgres_type(columns.paper_ids), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.comparison_report_json), postgresql.JSONB)


def test_workflow_job_table_matches_async_job_contract_columns():
    columns = WorkflowJobORM.__table__.c

    for name in [
        "session_id",
        "kind",
        "status",
        "input_json",
        "result_json",
        "error_json",
        "attempts",
        "max_attempts",
        "idempotency_key",
        "pipeline_version",
        "next_attempt_at",
        "retry_policy_json",
        "locked_by",
        "locked_at",
        "lease_expires_at",
        "heartbeat_at",
        "cancel_requested_at",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert isinstance(_postgres_type(columns.input_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.result_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.error_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.retry_policy_json), postgresql.JSONB)
    assert {
        "ck_workflow_jobs_kind",
        "ck_workflow_jobs_status",
        "ck_workflow_jobs_attempts_nonnegative",
        "ck_workflow_jobs_max_attempts_positive",
    }.issubset({constraint.name for constraint in WorkflowJobORM.__table__.constraints})
    assert {
        "ix_workflow_jobs_session_created_at",
        "ix_workflow_jobs_status_created_at",
        "ix_workflow_jobs_kind_status",
    }.issubset({index.name for index in WorkflowJobORM.__table__.indexes})


def test_arxiv_metadata_cache_table_matches_resilience_contract_columns():
    columns = ArxivMetadataCacheORM.__table__.c

    for name in [
        "arxiv_id",
        "title",
        "authors_json",
        "abstract",
        "published_date",
        "categories_json",
        "source_url",
        "fetched_at",
        "last_error_json",
        "error_count",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.arxiv_id.primary_key
    assert not ArxivMetadataCacheORM.__table__.foreign_keys
    assert isinstance(_postgres_type(columns.authors_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.categories_json), postgresql.JSONB)
    assert isinstance(_postgres_type(columns.last_error_json), postgresql.JSONB)
    assert "ck_arxiv_metadata_cache_error_count_nonnegative" in {
        constraint.name for constraint in ArxivMetadataCacheORM.__table__.constraints
    }
    assert "ix_arxiv_metadata_cache_fetched_at" in {
        index.name for index in ArxivMetadataCacheORM.__table__.indexes
    }


def test_blob_artifact_table_matches_registry_contract_columns():
    columns = BlobArtifactORM.__table__.c

    for name in [
        "id",
        "kind",
        "object_key",
        "bucket_name",
        "content_hash",
        "content_type",
        "size_bytes",
        "storage_backend",
        "retention_policy",
        "status",
        "expires_at",
        "last_accessed_at",
        "deleted_at",
        "cleanup_metadata_json",
        "created_at",
        "updated_at",
    ]:
        assert name in columns

    assert columns.id.primary_key
    assert {
        "uq_blob_artifacts_kind_content_hash",
        "ck_blob_artifacts_kind",
        "ck_blob_artifacts_retention_policy",
        "ck_blob_artifacts_retention_expiry",
        "ck_blob_artifacts_size_nonnegative",
        "ck_blob_artifacts_status",
        "ck_blob_artifacts_deletion_state",
    }.issubset({constraint.name for constraint in BlobArtifactORM.__table__.constraints})


def test_blob_reference_table_matches_polymorphic_reference_contract():
    columns = BlobReferenceORM.__table__.c

    for name in [
        "id", "blob_id", "ref_kind", "ref_id", "metadata_json",
        "status", "released_at", "created_at",
    ]:
        assert name in columns

    foreign_keys = {fk.target_fullname for fk in BlobReferenceORM.__table__.foreign_keys}
    assert foreign_keys == {"blob_artifacts.id"}
    assert isinstance(_postgres_type(columns.metadata_json), postgresql.JSONB)
    assert {
        "uq_blob_references_blob_kind_ref",
        "ck_blob_references_ref_kind",
        "ck_blob_references_status",
        "ck_blob_references_release_state",
    }.issubset({constraint.name for constraint in BlobReferenceORM.__table__.constraints})
    assert "ix_blob_references_kind_ref" in {
        index.name for index in BlobReferenceORM.__table__.indexes
    }


def test_pdf_upload_table_matches_durable_upload_contract():
    columns = PdfUploadORM.__table__.c

    for name in [
        "id", "session_id", "blob_id", "object_key", "expected_sha256",
        "actual_sha256", "size_bytes", "content_type", "status",
        "expires_at", "finalized_at", "error_json", "created_at", "updated_at",
    ]:
        assert name in columns

    foreign_keys = {fk.target_fullname for fk in PdfUploadORM.__table__.foreign_keys}
    assert foreign_keys == {"sessions.id", "blob_artifacts.id"}
    assert isinstance(_postgres_type(columns.error_json), postgresql.JSONB)
    assert {
        "ck_pdf_uploads_status",
        "ck_pdf_uploads_size_nonnegative",
        "ck_pdf_uploads_finalized_integrity",
        "uq_pdf_uploads_object_key",
    }.issubset({constraint.name for constraint in PdfUploadORM.__table__.constraints})
    assert "ix_pdf_uploads_session_created_at" in {
        index.name for index in PdfUploadORM.__table__.indexes
    }
