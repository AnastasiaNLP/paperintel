import logging
from datetime import timezone
from typing import Protocol

from models.agent_runs import AgentRun, TerminationReason
from services.observability import emit_event


logger = logging.getLogger(__name__)


class AgentRunRecorder(Protocol):
    def start(
        self,
        *,
        agent_name: str,
        session_id: str | None = None,
        job_id: str | None = None,
        input_refs: list[str] | None = None,
        model: str | None = None,
        iteration_count: int = 0,
    ) -> AgentRun:
        ...

    def complete(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        confidence: float | None = None,
        termination_reason: TerminationReason = "success",
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        details: dict | None = None,
    ) -> AgentRun:
        ...

    def fail(
        self,
        run_id: str,
        *,
        termination_reason: TerminationReason = "error",
        output_ref: str | None = None,
        details: dict | None = None,
    ) -> AgentRun:
        ...

    def fallback(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        termination_reason: TerminationReason = "fallback",
        details: dict | None = None,
    ) -> AgentRun:
        ...

    def get(self, run_id: str) -> AgentRun:
        ...

    def list_runs(self) -> list[AgentRun]:
        ...


class AgentRunPersistence(Protocol):
    def save(self, run: AgentRun) -> None:
        ...


class NoopAgentRunPersistence:
    def __init__(self) -> None:
        self._observed_started: set[str] = set()
        self._observed_terminal: set[str] = set()

    def save(self, run: AgentRun) -> None:
        emit_agent_run_lifecycle_events(
            run,
            observed_started=self._observed_started,
            observed_terminal=self._observed_terminal,
        )
        return None


class InMemoryAgentRunPersistence:
    def __init__(self) -> None:
        self._runs: list[AgentRun] = []
        self._observed_started: set[str] = set()
        self._observed_terminal: set[str] = set()

    def save(self, run: AgentRun) -> None:
        emit_agent_run_lifecycle_events(
            run,
            observed_started=self._observed_started,
            observed_terminal=self._observed_terminal,
        )
        self._runs.append(run)

    def list_runs(self) -> list[AgentRun]:
        return list(self._runs)


class InMemoryAgentRunRecorder:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    def start(
        self,
        *,
        agent_name: str,
        session_id: str | None = None,
        job_id: str | None = None,
        input_refs: list[str] | None = None,
        model: str | None = None,
        iteration_count: int = 0,
    ) -> AgentRun:
        run = AgentRun(
            agent_name=agent_name,
            session_id=session_id,
            job_id=job_id,
            input_refs=input_refs or [],
            model=model,
            iteration_count=iteration_count,
        )
        self._runs[run.id] = run
        return run

    def complete(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        confidence: float | None = None,
        termination_reason: TerminationReason = "success",
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        details: dict | None = None,
    ) -> AgentRun:
        run = self.get(run_id)
        return run.complete(
            output_ref=output_ref,
            confidence=confidence,
            termination_reason=termination_reason,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            details=details,
        )

    def fail(
        self,
        run_id: str,
        *,
        termination_reason: TerminationReason = "error",
        output_ref: str | None = None,
        details: dict | None = None,
    ) -> AgentRun:
        return self.get(run_id).fail(
            termination_reason=termination_reason,
            output_ref=output_ref,
            details=details,
        )

    def fallback(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        termination_reason: TerminationReason = "fallback",
        details: dict | None = None,
    ) -> AgentRun:
        return self.get(run_id).fallback(
            output_ref=output_ref,
            termination_reason=termination_reason,
            details=details,
        )

    def get(self, run_id: str) -> AgentRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"AgentRun not found: {run_id}") from exc

    def list_runs(self) -> list[AgentRun]:
        return list(self._runs.values())


def emit_agent_run_lifecycle_events(
    run: AgentRun,
    *,
    observed_started: set[str],
    observed_terminal: set[str],
) -> None:
    if run.status == "running":
        if run.id not in observed_started:
            emit_agent_run_started(run)
        observed_started.add(run.id)
        return

    if run.id in observed_terminal:
        return
    observed_terminal.add(run.id)
    _emit_agent_run_terminal(run)


def emit_agent_run_started(run: AgentRun) -> None:
    emit_event(
        logger,
        "agent.started",
        agent_run_id=run.id,
        agent_name=run.agent_name,
        session_id=run.session_id,
        job_id=run.job_id,
        model=run.model,
    )


def _emit_agent_run_terminal(run: AgentRun) -> None:
    event = "agent.completed" if run.status == "completed" else "agent.failed"
    emit_event(
        logger,
        event,
        agent_run_id=run.id,
        agent_name=run.agent_name,
        session_id=run.session_id,
        job_id=run.job_id,
        model=run.model,
        status=run.status,
        termination_reason=run.termination_reason,
        failure_class=(
            _agent_failure_class(run)
            if event == "agent.failed"
            else None
        ),
        duration_ms=_agent_duration_ms(run),
    )


def _agent_failure_class(run: AgentRun) -> str:
    return run.termination_reason or "error"


def _agent_duration_ms(run: AgentRun) -> int | None:
    if run.finished_at is None:
        return None
    started = run.started_at
    finished = run.finished_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    return max(0, round((finished - started).total_seconds() * 1000))
