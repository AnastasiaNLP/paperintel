import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from models.agent_runs import AgentRun
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.errors import ErrorCodes, make_error
from models.state import PaperIntelState

logger = logging.getLogger(__name__)


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _agent_run_persistence(config: RunnableConfig | None) -> AgentRunPersistence:
    configurable = _configurable(config)
    persistence = configurable.get("agent_run_persistence")
    if persistence is None:
        persistence = (
            config.get("agent_run_persistence") if isinstance(config, dict) else None
        )
    if persistence is not None and hasattr(persistence, "save"):
        return persistence
    return NoopAgentRunPersistence()


def _latest_report_run_id(state: PaperIntelState) -> str | None:
    for run in reversed(state.get("agent_runs", []) or []):
        if getattr(run, "agent_name", None) == "report":
            return getattr(run, "id", None)
    return None


def _start_critic_run(
    state: PaperIntelState,
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    input_refs = ["state:report"]
    report_run_id = _latest_report_run_id(state)
    if report_run_id:
        input_refs.append(report_run_id)

    return AgentRun(
        agent_name="evidence_critic",
        session_id=configurable.get("session_id"),
        job_id=configurable.get("job_id"),
        input_refs=input_refs,
        iteration_count=1,
    )


def _policy_snapshot(policy: AgentRuntimePolicy) -> dict[str, Any]:
    return policy.model_dump(mode="json")


def _with_agent_run(
    result: dict,
    run: AgentRun,
    persistence: AgentRunPersistence,
) -> dict:
    persistence.save(run)
    result["agent_runs"] = [run]
    return result


def _warning(message: str):
    return make_error(
        ErrorCodes.WARNING,
        message,
        node="evidence_critic",
        agent="evidence_critic",
        severity="warning",
        recoverable=True,
    )


def _append_reason(existing: str, note: str) -> str:
    existing = existing.strip()
    if not existing:
        return note
    if note.lower() in existing.lower():
        return existing
    return f"{existing} {note}"


def evidence_critic_agent(
    state: PaperIntelState,
    config: RunnableConfig | None = None,
) -> dict:
    """
    Minimal Evidence Critic.

    This first version is intentionally conservative and deterministic:
    it does not call an LLM, does not block finalization, and only downgrades
    obviously overconfident report recommendations when upstream evidence is
    missing or production maturity is weak.
    """
    run = _start_critic_run(state, config)
    persistence = _agent_run_persistence(config)
    policy = resolve_agent_policy("evidence_critic", config)
    run.details["policy_applied"] = _policy_snapshot(policy)

    engineer_report = state.get("engineer_report")
    if engineer_report is None:
        run.complete(
            output_ref="state:report",
            termination_reason="skipped",
            details={
                "reason": "no_report_to_review",
                "fallback_used": True,
                "fallback_reason": "no_report_to_review",
            },
        )
        return _with_agent_run({}, run, persistence)

    readiness = state.get("production_readiness")
    benchmarks = state.get("benchmarks", []) or []
    errors = []
    changed = False

    updated_report = engineer_report.model_copy(deep=True)

    if not benchmarks and updated_report.recommended_action == "implement_now":
        updated_report.recommended_action = "prototype"
        updated_report.action_reasoning = _append_reason(
            updated_report.action_reasoning,
            "Evidence Critic downgraded from implement_now because no benchmarks were extracted.",
        )
        errors.append(
            _warning(
                "Evidence Critic downgraded report: implement_now without benchmark evidence"
            )
        )
        changed = True

    if readiness is None and updated_report.recommended_action in {
        "implement_now",
        "prototype",
    }:
        original = updated_report.recommended_action
        updated_report.recommended_action = "watch"
        updated_report.implementation_difficulty = "research_only"
        updated_report.action_reasoning = _append_reason(
            updated_report.action_reasoning,
            (
                f"Evidence Critic downgraded from {original} because production "
                "readiness evidence is unavailable."
            ),
        )
        errors.append(
            _warning(
                "Evidence Critic downgraded report: production readiness unavailable"
            )
        )
        changed = True

    if (
        readiness is not None
        and readiness.maturity_level == "research_only"
        and updated_report.recommended_action in {"implement_now", "prototype"}
    ):
        original = updated_report.recommended_action
        updated_report.recommended_action = "watch"
        updated_report.implementation_difficulty = "research_only"
        updated_report.action_reasoning = _append_reason(
            updated_report.action_reasoning,
            (
                f"Evidence Critic downgraded from {original} because maturity "
                "is research_only."
            ),
        )
        errors.append(
            _warning("Evidence Critic downgraded report: maturity is research_only")
        )
        changed = True

    if changed:
        logger.warning(
            "Evidence Critic adjusted report action=%s difficulty=%s",
            updated_report.recommended_action,
            updated_report.implementation_difficulty,
        )
        run.complete(
            output_ref="state:report",
            details={
                "reviewed": True,
                "changed": True,
                "warnings_count": len(errors),
            },
        )
        return _with_agent_run(
            {
                "engineer_report": updated_report,
                "errors": errors,
            },
            run,
            persistence,
        )

    logger.info("Evidence Critic accepted report without changes")
    run.complete(
        output_ref="state:report",
        details={
            "reviewed": True,
            "changed": False,
            "warnings_count": 0,
        },
    )
    return _with_agent_run({}, run, persistence)
