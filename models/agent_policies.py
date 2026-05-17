from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class UnknownAgentPolicyError(KeyError):
    pass


class AgentRuntimePolicy(BaseModel):
    max_iterations: int = Field(default=1, ge=0)
    max_tool_calls: int = Field(default=0, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    timeout_seconds: int | None = Field(default=None, ge=0)
    fallback_strategy: str


DEFAULT_AGENT_POLICIES: dict[str, AgentRuntimePolicy] = {
    "report": AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=2,
        max_tokens=12_000,
        timeout_seconds=90,
        fallback_strategy="repair_on_invalid_json",
    ),
    "evidence_critic": AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=0,
        max_tokens=None,
        timeout_seconds=None,
        fallback_strategy="skip_review_on_no_report",
    ),
    "answer_agent": AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=4_000,
        timeout_seconds=60,
        fallback_strategy="insufficient_evidence_response",
    ),
    "citation_critic": AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=1,
        max_tokens=3_000,
        timeout_seconds=60,
        fallback_strategy="downgrade_after_repair_exhaustion",
    ),
    "intent_router": AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=1_500,
        timeout_seconds=15,
        fallback_strategy="ask_clarification",
    ),
    "retrieval_planner": AgentRuntimePolicy(
        max_iterations=2,
        max_tool_calls=4,
        max_tokens=2_000,
        timeout_seconds=30,
        fallback_strategy="return_best_effort_evidence",
    ),
    "research_strategist": AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=1_500,
        timeout_seconds=20,
        fallback_strategy="single_query_fallback",
    ),
}

CONSERVATIVE_AGENT_POLICY = AgentRuntimePolicy(
    max_iterations=1,
    max_tool_calls=0,
    max_tokens=None,
    timeout_seconds=None,
    fallback_strategy="strict_default",
)


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _policy_overrides(config: RunnableConfig | None) -> dict[str, AgentRuntimePolicy]:
    overrides = _configurable(config).get("agent_policy_overrides")
    return overrides if isinstance(overrides, dict) else {}


def _coerce_policy(value: AgentRuntimePolicy | dict[str, Any]) -> AgentRuntimePolicy:
    if isinstance(value, AgentRuntimePolicy):
        return value
    if isinstance(value, dict):
        return AgentRuntimePolicy.model_validate(value)
    raise TypeError(f"Invalid AgentRuntimePolicy override: {type(value).__name__}")


def resolve_agent_policy(
    agent_name: str,
    config: RunnableConfig | None = None,
    *,
    strict: bool = True,
) -> AgentRuntimePolicy:
    overrides = _policy_overrides(config)
    if agent_name in overrides:
        return _coerce_policy(overrides[agent_name])

    policy = DEFAULT_AGENT_POLICIES.get(agent_name)
    if policy is not None:
        return policy

    if strict:
        raise UnknownAgentPolicyError(f"No AgentRuntimePolicy registered for {agent_name}")

    return CONSERVATIVE_AGENT_POLICY
