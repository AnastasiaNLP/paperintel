import logging

from models.errors import ErrorCodes, make_error
from models.state import PaperIntelState

logger = logging.getLogger(__name__)


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


def evidence_critic_agent(state: PaperIntelState) -> dict:
    """
    Minimal Evidence Critic.

    This first version is intentionally conservative and deterministic:
    it does not call an LLM, does not block finalization, and only downgrades
    obviously overconfident report recommendations when upstream evidence is
    missing or production maturity is weak.
    """
    engineer_report = state.get("engineer_report")
    if engineer_report is None:
        return {}

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
        return {
            "engineer_report": updated_report,
            "errors": errors,
        }

    logger.info("Evidence Critic accepted report without changes")
    return {}
