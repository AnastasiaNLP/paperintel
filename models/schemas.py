from typing import List, Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    arxiv_id: Optional[str] = None
    published_date: str
    abstract: str
    categories: List[str]
    citation_count: Optional[int] = None


class MethodExtraction(BaseModel):
    method_name: str
    description: str
    novelty_claim: str
    key_components: List[str]
    compared_to: List[str]
    limitations_stated: List[str]


class BenchmarkResult(BaseModel):
    task: str
    metric: str
    value: float
    unit: Optional[str] = None
    baseline_comparison: Optional[str] = None
    conditions: Optional[str] = None


class ProductionReadiness(BaseModel):
    has_open_code: bool
    code_url: Optional[str] = None
    huggingface_model: Optional[str] = None
    framework_integrations: List[str]
    min_gpu_requirement: Optional[str] = None
    estimated_inference_cost: Optional[str] = None
    dependencies: List[str]
    maturity_level: str  # "research_only" | "experimental" | "production_ready"
    maturity_reasoning: str


class EngineerReport(BaseModel):
    executive_summary: str
    key_innovation: str
    practical_implications: str
    implementation_difficulty: str  # "trivial" | "moderate" | "significant" | "research_only"
    recommended_action: str  # "implement_now" | "prototype" | "watch" | "skip"
    action_reasoning: str


class PaperSlot(BaseModel):
    """
    One paper's finalized state inside a multi-paper comparison session.

    Populated after the single-paper pipeline completes. Comparator reads a
    collection of PaperSlot objects and never needs raw PDF text.
    """

    paper_index: int
    input_url: str

    metadata: Optional[PaperMetadata] = None
    method_extraction: Optional[MethodExtraction] = None
    benchmarks: List[BenchmarkResult] = Field(default_factory=list)
    production_readiness: Optional[ProductionReadiness] = None
    engineer_report: Optional[EngineerReport] = None
    markdown_report: Optional[str] = None

    errors: List[str] = Field(default_factory=list)
    completed: bool = False


class ComparisonMatrixRow(BaseModel):
    """
    One deterministic benchmark alignment row across compared papers.

    Missing values are explicit None entries so downstream UI and LLM prompts can
    distinguish "not reported" from an absent paper key.
    """

    task: str
    metric: str
    values_by_paper: dict[int, Optional[float]]
    units_by_paper: dict[int, Optional[str]] = Field(default_factory=dict)
    conditions_by_paper: dict[int, Optional[str]] = Field(default_factory=dict)
    duplicate_counts_by_paper: dict[int, int] = Field(default_factory=dict)
    winner_index: Optional[int] = None
    winner_margin: Optional[float] = None
    higher_is_better: bool = True
    is_comparable: bool = True
    comparability_notes: Optional[str] = None


class ConstraintRecommendation(BaseModel):
    """
    Comparator recommendation for a specific production constraint.
    """

    constraint: str
    recommended_paper_index: int
    reasoning: str


class ComparisonReport(BaseModel):
    """
    Structured Comparator output.

    The markdown comparison can be rendered separately from this schema for UI
    and report display.
    """

    papers_summary: List[dict] = Field(default_factory=list)
    comparison_matrix: List[ComparisonMatrixRow] = Field(default_factory=list)
    unique_tasks_per_paper: dict[int, List[str]] = Field(default_factory=dict)
    unique_rows_per_paper: dict[int, List[str]] = Field(default_factory=dict)
    comparable_rows: int = 0
    rows_with_winner: int = 0
    benchmark_overlap_ratio: float = 0.0
    wins_by_paper: dict[int, int] = Field(default_factory=dict)
    winner_basis: str = "no_clear_winner"
    trade_offs: str
    recommendations: List[ConstraintRecommendation] = Field(default_factory=list)
    overall_winner_index: Optional[int] = None
    overall_winner_reasoning: str
