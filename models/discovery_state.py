from operator import add as add_lists
from typing import Annotated, TypedDict

from models.agent_runs import AgentRun
from models.discovery import DiscoveryPlan, SearchCandidate, SelectionAdvice
from models.errors import StructuredError
from models.session import Persona, SessionPhase


class DiscoveryState(TypedDict, total=False):
    session_id: str
    user_message: str
    persona: Persona
    discovery_turn_id: str

    discovery_topic: str
    discovery_plan: DiscoveryPlan

    search_candidates: list[SearchCandidate]
    search_warnings: list[str]

    selection_advice: SelectionAdvice
    response_text: str
    next_phase: SessionPhase

    agent_runs: Annotated[list[AgentRun], add_lists]
    errors: Annotated[list[StructuredError], add_lists]
