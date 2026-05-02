from typing import Protocol

from models.agent_runs import AgentRun, TerminationReason


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
    ) -> AgentRun:
        ...

    def fail(
        self,
        run_id: str,
        *,
        termination_reason: TerminationReason = "error",
        output_ref: str | None = None,
    ) -> AgentRun:
        ...

    def fallback(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        termination_reason: TerminationReason = "fallback",
    ) -> AgentRun:
        ...

    def get(self, run_id: str) -> AgentRun:
        ...

    def list_runs(self) -> list[AgentRun]:
        ...


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
    ) -> AgentRun:
        run = self.get(run_id)
        return run.complete(
            output_ref=output_ref,
            confidence=confidence,
            termination_reason=termination_reason,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

    def fail(
        self,
        run_id: str,
        *,
        termination_reason: TerminationReason = "error",
        output_ref: str | None = None,
    ) -> AgentRun:
        return self.get(run_id).fail(
            termination_reason=termination_reason,
            output_ref=output_ref,
        )

    def fallback(
        self,
        run_id: str,
        *,
        output_ref: str | None = None,
        termination_reason: TerminationReason = "fallback",
    ) -> AgentRun:
        return self.get(run_id).fallback(
            output_ref=output_ref,
            termination_reason=termination_reason,
        )

    def get(self, run_id: str) -> AgentRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"AgentRun not found: {run_id}") from exc

    def list_runs(self) -> list[AgentRun]:
        return list(self._runs.values())
