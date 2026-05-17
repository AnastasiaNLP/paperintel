from sqlalchemy.dialects import postgresql

from storage.models import (
    AgentRunORM,
    Base,
    PaperChunkORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
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
