from dataclasses import dataclass

from pydantic import BaseModel, Field

from models.agent_runs import AgentRun
from models.session import Persona


class SynthesisCitation(BaseModel):
    paper_id: str
    quote_or_summary: str


class SynthesisRecommendation(BaseModel):
    recommendation: str
    reasoning: str


class SynthesisReport(BaseModel):
    persona: Persona
    summary: str
    key_takeaways: list[str]
    trade_offs: list[str]
    recommended_next_steps: list[SynthesisRecommendation]
    citations: list[SynthesisCitation]
    limitations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SynthesisAgentResult:
    report: SynthesisReport
    response_text: str
    agent_run: AgentRun
