from typing import TypedDict, List, Optional, Annotated, Literal
from langgraph.graph.message import add_messages
from models.schemas import (
    PaperMetadata,
    MethodExtraction,
    BenchmarkResult,
    ProductionReadiness,
    EngineerReport,
)


def add_to_list(existing: list, new: list) -> list:
    return existing + new


ProcessingStage = Literal[
    "ingestion",
    "extraction",
    "benchmark",
    "readiness",
    "report",
    "topic_selection",
    "failed",
]


class IngestionProvenance(TypedDict):
    """
    Provenance metadata from the Ingestion Agent.
    Downstream agents use this to understand input quality.
    """
    text_source: Literal["pdf", "abstract_fallback", "none"]
    metadata_source: Literal["arxiv", "pdf_fallback", "none"]
    enrichment_status: Literal["s2_ok", "s2_failed", "not_attempted"]
    arxiv_id_found: bool


class PaperIntelState(TypedDict):
    # Input
    input_type: str        # "url" | "pdf" | "topic_query"
    input_value: str       # URL, file path, or search query

    # Paper data
    papers: Annotated[list, add_to_list]

    # Agent outputs
    metadata: Optional[PaperMetadata]
    raw_text: Optional[str]
    pdf_path: Optional[str]
    text_by_page: Optional[dict[int, str]]
    method_extraction: Optional[MethodExtraction]
    benchmarks: List[BenchmarkResult]
    production_readiness: Optional[ProductionReadiness]
    ingestion_provenance: Optional[IngestionProvenance]

    # Multi-paper
    comparison_report: Optional[str]

    # Final output
    engineer_report: Optional[EngineerReport]
    full_markdown_report: Optional[str]

    # Control flow
    current_paper_index: int
    total_papers: int
    processing_stage: ProcessingStage
    needs_human_review: bool
    human_review_reason: Optional[str]
    confidence_scores: dict

    # Metadata
    messages: Annotated[list, add_messages]
    errors: List[str]
    cost_tracking: dict
