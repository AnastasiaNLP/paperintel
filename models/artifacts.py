from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from models.session import utc_now


class PaperWorkspace(BaseModel):
    """
    Durable per-session artifact snapshot for one analyzed paper.

    paper_id matches the session.active_paper_ids value. For arXiv papers this
    is the arXiv ID; future non-arXiv sources should use their canonical active
    paper ID.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    paper_id: str
    title: str | None = None
    source_url: str
    pipeline_stage: str
    finalized_report_json: dict | None = None
    method_extraction_json: dict | None = None
    benchmarks_json: list[dict] = Field(default_factory=list)
    readiness_json: dict | None = None
    full_markdown_report: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ComparisonArtifact(BaseModel):
    """Durable session-scoped comparison artifact for a group of papers."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    paper_ids: list[str] = Field(default_factory=list)
    comparison_report_json: dict | None = None
    comparison_markdown: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
