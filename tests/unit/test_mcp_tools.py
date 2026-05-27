import asyncio

import pytest

import mcp_server.tools as tool_module
from mcp_server.tools import (
    analyze_paper_tool,
    analyze_pdf_tool,
    analyze_selected_papers_tool,
    cancel_workflow_job_tool,
    ask_paper_tool,
    compare_papers_tool,
    create_session_tool,
    enqueue_analyze_paper_tool,
    enqueue_analyze_selected_tool,
    discover_papers_tool,
    format_comparison_artifact,
    format_answer_result,
    format_discovery_result,
    format_paper_workspace,
    get_session_tool,
    get_workflow_job_tool,
    get_latest_comparison_tool,
    get_paper_workspace_tool,
    list_paper_workspaces_tool,
    list_workflow_jobs_tool,
    select_papers_tool,
    synthesize_papers_tool,
)
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.agent_runs import AgentRun
from models.jobs import WorkflowJob
from models.retrieval import CitationRef
from models.session import HandlerResult, Session
from models.synthesis import (
    SynthesisAgentResult,
    SynthesisCitation,
    SynthesisRecommendation,
    SynthesisReport,
)


class FakeService:
    def __init__(self) -> None:
        self.sessions = {
            "session-1": Session(
                id="session-1",
                persona="engineer",
                phase="qa",
                active_paper_ids=["1706.03762"],
            )
        }
        self.create_calls = []
        self.analyze_calls = []
        self.analyze_pdf_calls = []
        self.ask_calls = []
        self.discover_calls = []
        self.select_calls = []
        self.analyze_selected_calls = []
        self.synthesize_calls = []
        self.compare_calls = []
        self.list_workspace_calls = []
        self.get_workspace_calls = []
        self.comparison_calls = []
        self.enqueue_analyze_paper_calls = []
        self.enqueue_analyze_selected_calls = []
        self.workflow_job_calls = []
        self.workflow_job_list_calls = []
        self.workflow_job_cancel_calls = []
        self.jobs = [
            WorkflowJob(
                id="job-1",
                session_id="session-1",
                kind="analyze_paper",
                status="queued",
                input_json={"paper_url": "https://arxiv.org/abs/1706.03762"},
            )
        ]
        self.workspaces = [
            PaperWorkspace(
                session_id="session-1",
                paper_id="1706.03762",
                title="Attention Is All You Need",
                source_url="https://arxiv.org/abs/1706.03762",
                pipeline_stage="completed",
                method_extraction_json={
                    "method_name": "Transformer",
                    "novelty_claim": "Attention-only sequence model.",
                },
                benchmarks_json=[
                    {"task": "translation", "metric": "BLEU", "value": 28.4}
                ],
                readiness_json={
                    "maturity_level": "production",
                    "has_open_code": True,
                },
                full_markdown_report="# Report",
            )
        ]
        self.comparison = ComparisonArtifact(
            session_id="session-1",
            paper_ids=["1706.03762", "2401.00001"],
            comparison_markdown="# Comparison\n\nA vs B",
        )

    def create_session(self, *, persona="engineer", original_query=None):
        self.create_calls.append({"persona": persona, "original_query": original_query})
        return Session(id="created-session", persona=persona)

    def analyze_paper(self, session_id, paper_url):
        self.analyze_calls.append((session_id, paper_url))
        return HandlerResult(
            session_id=session_id,
            response_text="Analysis complete.",
            phase="qa",
            referenced_paper_ids=["1706.03762"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_pdf(
        self,
        session_id,
        pdf_path,
        *,
        paper_id=None,
        skip_arxiv_metadata_fetch=False,
    ):
        self.analyze_pdf_calls.append(
            {
                "session_id": session_id,
                "pdf_path": pdf_path,
                "paper_id": paper_id,
                "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
            }
        )
        return HandlerResult(
            session_id=session_id,
            response_text="PDF analysis complete.",
            phase="qa",
            referenced_paper_ids=[paper_id or "local-pdf"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def ask_question(self, session_id, question):
        self.ask_calls.append((session_id, question))
        return HandlerResult(
            session_id=session_id,
            response_text="The Transformer replaces recurrence with attention.",
            phase="qa",
            intent="qa_factual",
            citations=[
                CitationRef(
                    paper_id="1706.03762",
                    chunk_id="1706.03762:chunk:1",
                    page_start=1,
                    page_end=1,
                )
            ],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def discover_papers(self, session_id, topic):
        self.discover_calls.append((session_id, topic))
        return HandlerResult(
            session_id=session_id,
            response_text="Here are candidate papers. Reply with numbers.",
            phase="selection",
            intent="discover",
            discovery_topic="agent memory",
            discovery_candidate_count=3,
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def select_papers(self, session_id, selection):
        self.select_calls.append((session_id, selection))
        return HandlerResult(
            session_id=session_id,
            response_text="Selected papers 1 and 3.",
            phase="idle",
            intent="select_papers",
            selected_candidate_ids=["candidate-1", "candidate-3"],
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def analyze_selected_papers(self, session_id):
        self.analyze_selected_calls.append(session_id)
        return HandlerResult(
            session_id=session_id,
            response_text="Selected analysis complete.",
            phase="qa",
            intent="analyze_paper",
            referenced_paper_ids=["2605.1", "2605.3"],
            comparison_markdown="# Paper Comparison\n\n2605.1 vs 2605.3",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
        )

    def enqueue_analyze_paper(self, session_id, paper_url):
        self.enqueue_analyze_paper_calls.append((session_id, paper_url))
        return WorkflowJob(
            id="job-queued-paper",
            session_id=session_id,
            kind="analyze_paper",
            status="queued",
            input_json={"paper_url": paper_url},
        )

    def enqueue_analyze_selected(self, session_id):
        self.enqueue_analyze_selected_calls.append(session_id)
        return WorkflowJob(
            id="job-queued-selected",
            session_id=session_id,
            kind="analyze_selected",
            status="queued",
            input_json={},
        )

    def get_workflow_job(self, job_id):
        self.workflow_job_calls.append(job_id)
        return self.jobs[0].model_copy(update={"id": job_id})

    def list_workflow_jobs(self, session_id, *, limit=50):
        self.workflow_job_list_calls.append((session_id, limit))
        return self.jobs[:limit]

    def cancel_workflow_job(self, job_id):
        self.workflow_job_cancel_calls.append(job_id)
        return self.jobs[0].model_copy(update={"id": job_id, "status": "canceled"})

    def synthesize_papers(self, session_id, prompt=None):
        self.synthesize_calls.append((session_id, prompt))
        run = AgentRun(
            agent_name="synthesis_agent",
            session_id=session_id,
            input_refs=["paper_workspace:1706.03762"],
        )
        run.complete(output_ref="synthesis_report")
        return SynthesisAgentResult(
            report=SynthesisReport(
                persona="engineer",
                summary="The papers trade off quality and deployment cost.",
                key_takeaways=["Quality differs."],
                trade_offs=["Deployment cost differs."],
                recommended_next_steps=[
                    SynthesisRecommendation(
                        recommendation="Prototype the cheaper option.",
                        reasoning="It is lower risk.",
                    )
                ],
                citations=[
                    SynthesisCitation(
                        paper_id="1706.03762",
                        quote_or_summary="Transformer summary.",
                    )
                ],
            ),
            response_text="The papers trade off quality and deployment cost.",
            agent_run=run,
        )

    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        self.compare_calls.append((session_id, paper_ids, prompt))
        return ComparisonArtifact(
            session_id=session_id,
            paper_ids=paper_ids or ["1706.03762", "2401.00001"],
            comparison_report_json={"producer": "comparison_analyst"},
            comparison_markdown="# Comparison\n\nA vs B",
        )

    def get_session(self, session_id):
        return self.sessions[session_id]

    def list_paper_workspaces(self, session_id):
        self.list_workspace_calls.append(session_id)
        return self.workspaces

    def get_paper_workspace(self, session_id, paper_id):
        self.get_workspace_calls.append((session_id, paper_id))
        return self.workspaces[0]

    def get_latest_comparison(self, session_id):
        self.comparison_calls.append(session_id)
        return self.comparison


class ExplodingService(FakeService):
    def analyze_pdf(self, session_id, pdf_path, **kwargs):
        raise RuntimeError("internal details should not leak")

    def ask_question(self, session_id, question):
        raise RuntimeError("internal details should not leak")

    def analyze_selected_papers(self, session_id):
        raise RuntimeError("internal details should not leak")

    def synthesize_papers(self, session_id, prompt=None):
        raise RuntimeError("internal details should not leak")

    def compare_papers(self, session_id, paper_ids=None, prompt=None):
        raise RuntimeError("internal details should not leak")

    def get_paper_workspace(self, session_id, paper_id):
        raise RuntimeError("internal details should not leak")

    def get_latest_comparison(self, session_id):
        raise RuntimeError("internal details should not leak")

    def enqueue_analyze_paper(self, session_id, paper_url):
        raise RuntimeError("internal details should not leak")

    def get_workflow_job(self, job_id):
        raise RuntimeError("internal details should not leak")


@pytest.fixture(autouse=True)
def run_sync_inline(monkeypatch):
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(tool_module, "_run_sync", inline)


def test_create_session_tool_returns_session_id():
    service = FakeService()

    text = asyncio.run(create_session_tool(service, persona="researcher"))

    assert "Session ID: created-session" in text
    assert "Persona: researcher" in text
    assert "discover_papers" in text
    assert service.create_calls == [
        {"persona": "researcher", "original_query": None}
    ]


def test_create_session_rejects_invalid_persona():
    with pytest.raises(ValueError):
        asyncio.run(create_session_tool(FakeService(), persona="manager"))


def test_analyze_paper_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        analyze_paper_tool(
            service,
            session_id="session-1",
            paper_url="https://arxiv.org/abs/1706.03762",
        )
    )

    assert "Paper analysis completed." in text
    assert "1706.03762" in text
    assert service.analyze_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]


def test_analyze_paper_rejects_non_url():
    with pytest.raises(ValueError):
        asyncio.run(
            analyze_paper_tool(
                FakeService(),
                session_id="session-1",
                paper_url="arxiv 1706.03762",
            )
        )


def test_analyze_pdf_tool_calls_service_with_local_path():
    service = FakeService()

    text = asyncio.run(
        analyze_pdf_tool(
            service,
            session_id="session-1",
            pdf_path="/home/nastassia/Desktop/pdfs/1706.03762.pdf",
            paper_id="1706.03762",
            skip_arxiv_metadata_fetch=True,
        )
    )

    assert "Paper analysis completed." in text
    assert "PDF analysis complete." in text
    assert service.analyze_pdf_calls == [
        {
            "session_id": "session-1",
            "pdf_path": "/home/nastassia/Desktop/pdfs/1706.03762.pdf",
            "paper_id": "1706.03762",
            "skip_arxiv_metadata_fetch": True,
        }
    ]


def test_analyze_pdf_tool_treats_blank_paper_id_as_none():
    service = FakeService()

    asyncio.run(
        analyze_pdf_tool(
            service,
            session_id="session-1",
            pdf_path="/tmp/paper.pdf",
            paper_id="   ",
        )
    )

    assert service.analyze_pdf_calls[0]["paper_id"] is None


def test_analyze_pdf_tool_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        asyncio.run(
            analyze_pdf_tool(
                FakeService(),
                session_id="",
                pdf_path="/tmp/paper.pdf",
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            analyze_pdf_tool(
                FakeService(),
                session_id="session-1",
                pdf_path="   ",
            )
        )


def test_analyze_pdf_tool_handles_service_exception_safely():
    text = asyncio.run(
        analyze_pdf_tool(
            ExplodingService(),
            session_id="session-1",
            pdf_path="/tmp/paper.pdf",
        )
    )

    assert "could not analyze the local PDF safely" in text
    assert "internal details" not in text


def test_ask_paper_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        ask_paper_tool(
            service,
            session_id="session-1",
            question="What is the contribution?",
        )
    )

    assert "The Transformer replaces recurrence" in text
    assert "Sources:" in text
    assert service.ask_calls == [("session-1", "What is the contribution?")]


def test_ask_paper_rejects_empty_question():
    with pytest.raises(ValueError):
        asyncio.run(ask_paper_tool(FakeService(), session_id="session-1", question=""))


def test_ask_paper_rejects_too_long_question():
    with pytest.raises(ValueError):
        asyncio.run(
            ask_paper_tool(
                FakeService(),
                session_id="session-1",
                question="x" * 2001,
            )
        )


def test_discover_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        discover_papers_tool(
            service,
            session_id="session-1",
            topic="Find papers about agent memory",
        )
    )

    assert "Here are candidate papers" in text
    assert "Candidates found: 3" in text
    assert "Session phase: selection" in text
    assert service.discover_calls == [
        ("session-1", "Find papers about agent memory")
    ]


def test_discover_papers_rejects_empty_topic():
    with pytest.raises(ValueError):
        asyncio.run(
            discover_papers_tool(FakeService(), session_id="session-1", topic="")
        )


def test_select_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        select_papers_tool(
            service,
            session_id="session-1",
            selection="use 1 and 3",
        )
    )

    assert "Selected papers 1 and 3" in text
    assert "candidate-1" in text
    assert service.select_calls == [("session-1", "use 1 and 3")]


def test_select_papers_rejects_empty_selection():
    with pytest.raises(ValueError):
        asyncio.run(
            select_papers_tool(FakeService(), session_id="session-1", selection="")
        )


def test_analyze_selected_papers_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        analyze_selected_papers_tool(service, session_id="session-1")
    )

    assert "Paper analysis completed." in text
    assert "Selected analysis complete." in text
    assert "- 2605.1" in text
    assert "Batch comparison report:" in text
    assert "# Paper Comparison" in text
    assert service.analyze_selected_calls == ["session-1"]


def test_analyze_selected_papers_rejects_empty_session_id():
    with pytest.raises(ValueError):
        asyncio.run(analyze_selected_papers_tool(FakeService(), session_id=""))


def test_analyze_selected_papers_handles_service_exception_safely():
    text = asyncio.run(
        analyze_selected_papers_tool(ExplodingService(), session_id="session-1")
    )

    assert "could not analyze the selected papers safely" in text
    assert "internal details" not in text


def test_enqueue_analyze_paper_tool_calls_service():
    service = FakeService()

    text = asyncio.run(
        enqueue_analyze_paper_tool(
            service,
            session_id="session-1",
            paper_url="https://arxiv.org/abs/1706.03762",
        )
    )

    assert "Queued paper analysis job" in text
    assert "Job ID: job-queued-paper" in text
    assert "Status: queued" in text
    assert service.enqueue_analyze_paper_calls == [
        ("session-1", "https://arxiv.org/abs/1706.03762")
    ]


def test_enqueue_analyze_selected_tool_calls_service():
    service = FakeService()

    text = asyncio.run(enqueue_analyze_selected_tool(service, session_id="session-1"))

    assert "Queued selected-paper analysis job" in text
    assert "Job ID: job-queued-selected" in text
    assert service.enqueue_analyze_selected_calls == ["session-1"]


def test_workflow_job_tools_get_list_and_cancel():
    service = FakeService()

    listed = asyncio.run(list_workflow_jobs_tool(service, session_id="session-1", limit=5))
    loaded = asyncio.run(get_workflow_job_tool(service, job_id="job-1"))
    canceled = asyncio.run(cancel_workflow_job_tool(service, job_id="job-1"))

    assert "Workflow jobs:" in listed
    assert "job-1: analyze_paper / queued" in listed
    assert "Workflow job" in loaded
    assert "Kind: analyze_paper" in loaded
    assert "Canceled workflow job" in canceled
    assert "Status: canceled" in canceled
    assert service.workflow_job_list_calls == [("session-1", 5)]
    assert service.workflow_job_calls == ["job-1"]
    assert service.workflow_job_cancel_calls == ["job-1"]


def test_workflow_job_tools_validate_inputs():
    with pytest.raises(ValueError):
        asyncio.run(enqueue_analyze_paper_tool(FakeService(), session_id="", paper_url="https://x.test"))
    with pytest.raises(ValueError):
        asyncio.run(enqueue_analyze_paper_tool(FakeService(), session_id="session-1", paper_url="not url"))
    with pytest.raises(ValueError):
        asyncio.run(list_workflow_jobs_tool(FakeService(), session_id="session-1", limit=0))
    with pytest.raises(ValueError):
        asyncio.run(get_workflow_job_tool(FakeService(), job_id=""))


def test_workflow_job_tools_handle_service_exception_safely():
    enqueue_text = asyncio.run(
        enqueue_analyze_paper_tool(
            ExplodingService(),
            session_id="session-1",
            paper_url="https://arxiv.org/abs/1706.03762",
        )
    )
    get_text = asyncio.run(get_workflow_job_tool(ExplodingService(), job_id="job-1"))

    assert "could not enqueue paper analysis safely" in enqueue_text
    assert "could not load the workflow job safely" in get_text
    assert "internal details" not in enqueue_text
    assert "internal details" not in get_text


def test_synthesize_papers_tool_calls_service_with_prompt():
    service = FakeService()

    text = asyncio.run(
        synthesize_papers_tool(
            service,
            session_id="session-1",
            prompt="Compare implementation trade-offs.",
        )
    )

    assert "quality and deployment cost" in text
    assert "Sources:" in text
    assert service.synthesize_calls == [
        ("session-1", "Compare implementation trade-offs.")
    ]


def test_synthesize_papers_tool_calls_service_without_prompt():
    service = FakeService()

    text = asyncio.run(synthesize_papers_tool(service, session_id="session-1"))

    assert "quality and deployment cost" in text
    assert service.synthesize_calls == [("session-1", None)]


def test_synthesize_papers_tool_treats_blank_prompt_as_default():
    service = FakeService()

    text = asyncio.run(
        synthesize_papers_tool(service, session_id="session-1", prompt="   ")
    )

    assert "quality and deployment cost" in text
    assert service.synthesize_calls == [("session-1", None)]


def test_synthesize_papers_rejects_empty_session_id():
    with pytest.raises(ValueError):
        asyncio.run(synthesize_papers_tool(FakeService(), session_id=""))


def test_synthesize_papers_rejects_too_long_prompt():
    with pytest.raises(ValueError):
        asyncio.run(
            synthesize_papers_tool(
                FakeService(),
                session_id="session-1",
                prompt="x" * 2001,
            )
        )


def test_synthesize_papers_handles_service_exception_safely():
    text = asyncio.run(
        synthesize_papers_tool(ExplodingService(), session_id="session-1")
    )

    assert "could not synthesize the active papers safely" in text
    assert "internal details" not in text


def test_compare_papers_tool_calls_service_with_prompt_and_paper_ids():
    service = FakeService()

    text = asyncio.run(
        compare_papers_tool(
            service,
            session_id="session-1",
            paper_ids=["2401.00001", "1706.03762"],
            prompt="Prefer deployability.",
        )
    )

    assert "Latest persisted comparison" in text
    assert "# Comparison" in text
    assert service.compare_calls == [
        ("session-1", ["2401.00001", "1706.03762"], "Prefer deployability.")
    ]


def test_compare_papers_tool_calls_service_without_paper_ids():
    service = FakeService()

    text = asyncio.run(compare_papers_tool(service, session_id="session-1"))

    assert "A vs B" in text
    assert service.compare_calls == [("session-1", None, None)]


def test_compare_papers_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        asyncio.run(compare_papers_tool(FakeService(), session_id=""))
    with pytest.raises(ValueError):
        asyncio.run(
            compare_papers_tool(
                FakeService(),
                session_id="session-1",
                prompt="x" * 2001,
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            compare_papers_tool(
                FakeService(),
                session_id="session-1",
                paper_ids=["1706.03762", "  "],
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            compare_papers_tool(
                FakeService(),
                session_id="session-1",
                paper_ids=["1706.03762", "1706.03762"],
            )
        )


def test_compare_papers_handles_service_exception_safely():
    text = asyncio.run(compare_papers_tool(ExplodingService(), session_id="session-1"))

    assert "could not compare the selected papers safely" in text
    assert "internal details" not in text


def test_get_session_tool_returns_state():
    text = asyncio.run(get_session_tool(FakeService(), session_id="session-1"))

    assert "Session: session-1" in text
    assert "Persona: engineer" in text
    assert "- 1706.03762" in text


def test_list_paper_workspaces_tool_returns_readable_summary():
    service = FakeService()

    text = asyncio.run(list_paper_workspaces_tool(service, session_id="session-1"))

    assert "Persisted paper workspaces" in text
    assert "1706.03762" in text
    assert "Artifacts: report, method, 1 benchmark(s), readiness" in text
    assert service.list_workspace_calls == ["session-1"]


def test_get_paper_workspace_tool_returns_readable_summary_not_raw_json():
    service = FakeService()

    text = asyncio.run(
        get_paper_workspace_tool(
            service,
            session_id="session-1",
            paper_id="1706.03762",
        )
    )

    assert "Paper workspace: 1706.03762" in text
    assert "Method:" in text
    assert "Transformer" in text
    assert "Benchmarks:" in text
    assert "# Report" in text
    assert "{'method_name'" not in text
    assert service.get_workspace_calls == [("session-1", "1706.03762")]


def test_get_latest_comparison_tool_returns_markdown():
    service = FakeService()

    text = asyncio.run(get_latest_comparison_tool(service, session_id="session-1"))

    assert "Latest persisted comparison" in text
    assert "- 1706.03762" in text
    assert "# Comparison" in text
    assert service.comparison_calls == ["session-1"]


def test_get_paper_workspace_rejects_empty_paper_id():
    with pytest.raises(ValueError):
        asyncio.run(
            get_paper_workspace_tool(
                FakeService(),
                session_id="session-1",
                paper_id="",
            )
        )


def test_get_paper_workspace_handles_service_exception_safely():
    text = asyncio.run(
        get_paper_workspace_tool(
            ExplodingService(),
            session_id="session-1",
            paper_id="1706.03762",
        )
    )

    assert "could not load the paper workspace safely" in text
    assert "internal details" not in text


def test_get_latest_comparison_handles_service_exception_safely():
    text = asyncio.run(
        get_latest_comparison_tool(ExplodingService(), session_id="session-1")
    )

    assert "could not load the latest comparison safely" in text
    assert "internal details" not in text


def test_format_answer_result_includes_citations():
    result = HandlerResult(
        session_id="session-1",
        response_text="Answer.",
        phase="qa",
        citations=[
            CitationRef(
                paper_id="1706.03762",
                chunk_id="1706.03762:chunk:14",
                page_start=8,
                page_end=9,
            )
        ],
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )

    text = format_answer_result(result)

    assert "Sources:" in text
    assert "1706.03762, pages 8-9, chunk 1706.03762:chunk:14" in text


def test_format_discovery_result_includes_topic_and_count():
    result = HandlerResult(
        session_id="session-1",
        response_text="Choose papers.",
        phase="selection",
        discovery_topic="agent memory",
        discovery_candidate_count=5,
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
    )

    text = format_discovery_result(result)

    assert "Topic: agent memory" in text
    assert "Candidates found: 5" in text
    assert "select_papers" in text


def test_format_paper_workspace_includes_sections():
    text = format_paper_workspace(FakeService().workspaces[0])

    assert "Production readiness:" in text
    assert "Maturity: production" in text


def test_format_comparison_artifact_includes_paper_ids():
    text = format_comparison_artifact(FakeService().comparison)

    assert "Papers:" in text
    assert "- 2401.00001" in text


def test_tool_handles_service_exception_safely():
    text = asyncio.run(
        ask_paper_tool(
            ExplodingService(),
            session_id="session-1",
            question="What is the contribution?",
        )
    )

    assert "could not answer the question safely" in text
    assert "internal details" not in text
