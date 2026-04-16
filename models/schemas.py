from typing import List, Optional
from pydantic import BaseModel


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