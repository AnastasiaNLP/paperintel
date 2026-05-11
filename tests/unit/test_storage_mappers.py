from models.agent_runs import AgentRun
from models.errors import ErrorCodes, StructuredError, make_error
from models.retrieval import ChunkLocation, ChunkSource, EvidenceArtifact, PaperChunk
from models.session import Session, Turn
from storage.mappers import (
    agent_run_to_orm,
    orm_to_agent_run,
    orm_to_paper_chunk,
    orm_to_session,
    orm_to_structured_error,
    orm_to_turn,
    paper_chunk_to_orm,
    session_to_orm,
    structured_error_to_orm,
    turn_to_orm,
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
