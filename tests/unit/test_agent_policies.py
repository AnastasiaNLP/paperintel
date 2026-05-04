import pytest

from models.agent_policies import (
    CONSERVATIVE_AGENT_POLICY,
    DEFAULT_AGENT_POLICIES,
    AgentRuntimePolicy,
    UnknownAgentPolicyError,
    resolve_agent_policy,
)


def test_default_policy_resolves_for_report():
    policy = resolve_agent_policy("report")

    assert policy.max_iterations == 1
    assert policy.max_tool_calls == 2
    assert policy.fallback_strategy == "repair_on_invalid_json"


def test_full_override_via_runnable_config_replaces_default():
    override = AgentRuntimePolicy(
        max_iterations=1,
        max_tool_calls=1,
        max_tokens=500,
        timeout_seconds=10,
        fallback_strategy="no_repair",
    )
    config = {"configurable": {"agent_policy_overrides": {"report": override}}}

    assert resolve_agent_policy("report", config) is override


def test_dict_override_via_runnable_config_is_validated():
    config = {
        "configurable": {
            "agent_policy_overrides": {
                "evidence_critic": {
                    "max_iterations": 1,
                    "max_tool_calls": 0,
                    "max_tokens": None,
                    "timeout_seconds": None,
                    "fallback_strategy": "skip_review_on_no_report",
                }
            }
        }
    }

    policy = resolve_agent_policy("evidence_critic", config)

    assert isinstance(policy, AgentRuntimePolicy)
    assert policy.fallback_strategy == "skip_review_on_no_report"


def test_unknown_agent_policy_strict_raises():
    with pytest.raises(UnknownAgentPolicyError, match="not_registered"):
        resolve_agent_policy("not_registered")


def test_unknown_agent_policy_non_strict_uses_conservative_default():
    assert resolve_agent_policy("not_registered", strict=False) == CONSERVATIVE_AGENT_POLICY


def test_current_pipeline_agent_policies_are_registered():
    expected = {"report", "evidence_critic"}

    assert expected.issubset(DEFAULT_AGENT_POLICIES.keys())
