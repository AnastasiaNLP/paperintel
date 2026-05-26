from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from agents.comparator import (
    _build_comparison_matrix,
    _build_evidence_json,
    _build_matrix_stats,
    _build_papers_summary,
    _build_unique_rows_per_paper,
    _build_unique_tasks_per_paper,
    _normalize_comparison_report,
    _parse_claims,
    _render_comparison_markdown,
)
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.agent_runs import AgentRun
from models.artifacts import PaperWorkspace
from models.schemas import (
    BenchmarkResult,
    ComparisonReport,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Comparison Analyst for AI research papers.

Return ONLY a JSON object. No markdown, no prose.

Use the provided durable PaperWorkspace evidence to compare papers. Ground the
comparison in extracted method, benchmark, readiness, and report fields. Do not
invent paper ids, benchmark values, code availability, or readiness claims.

Expected JSON shape:
{
  "trade_offs": "string",
  "recommendations": [
    {
      "constraint": "string",
      "recommended_paper_index": 0,
      "reasoning": "string"
    }
  ],
  "overall_winner_index": 0,
  "overall_winner_reasoning": "string",
  "winner_basis": "readiness_dominant | benchmark_dominant | mixed | no_clear_winner"
}

If evidence is weak or papers are not directly comparable, set
overall_winner_index to null and winner_basis to "no_clear_winner".
"""


@dataclass(frozen=True)
class ComparisonAnalystResult:
    report: ComparisonReport
    markdown: str
    agent_run: AgentRun


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _agent_run_persistence(config: RunnableConfig | None) -> AgentRunPersistence:
    configurable = _configurable(config)
    persistence = configurable.get("agent_run_persistence")
    if persistence is not None and hasattr(persistence, "save"):
        return persistence
    return NoopAgentRunPersistence()


def _policy_snapshot(policy: AgentRuntimePolicy) -> dict[str, Any]:
    return policy.model_dump(mode="json")


def _start_run(
    *,
    session_id: str,
    workspaces: list[PaperWorkspace],
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    return AgentRun(
        agent_name="comparison_analyst",
        session_id=session_id,
        job_id=configurable.get("job_id"),
        input_refs=[f"paper_workspace:{workspace.paper_id}" for workspace in workspaces],
        model=settings.sonnet_model,
        iteration_count=1,
    )


def _call_llm(
    evidence_json: str,
    *,
    policy: AgentRuntimePolicy,
    prompt: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Single-pass LLM reasoning for request-driven comparison.

    Unlike the batch comparator, this agent does not issue a JSON repair call.
    The deterministic comparison report is a useful fallback by itself, and
    avoiding repair keeps the request-driven path to one LLM call under the
    declared runtime policy.
    """
    user_content = evidence_json
    if prompt and prompt.strip():
        user_content += "\n\nUser comparison prompt:\n" + prompt.strip()
    return call_text_llm(
        requested_model=settings.sonnet_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=policy.max_tokens or 4_000,
        context_label="Comparison Analyst LLM",
    )


def _workspace_to_slot(workspace: PaperWorkspace, index: int) -> PaperSlot:
    return PaperSlot(
        paper_index=index,
        input_url=workspace.source_url,
        metadata=_metadata_from_workspace(workspace),
        method_extraction=_model_or_none(
            MethodExtraction,
            workspace.method_extraction_json,
        ),
        benchmarks=[
            benchmark
            for item in workspace.benchmarks_json
            if (benchmark := _model_or_none(BenchmarkResult, item)) is not None
        ],
        production_readiness=_model_or_none(
            ProductionReadiness,
            workspace.readiness_json,
        ),
        engineer_report=_model_or_none(EngineerReport, workspace.finalized_report_json),
        markdown_report=workspace.full_markdown_report,
        completed=workspace.pipeline_stage not in {"failed", "paper_failure_finalize"},
        errors=[],
    )


def _metadata_from_workspace(workspace: PaperWorkspace) -> PaperMetadata:
    return PaperMetadata(
        title=workspace.title or workspace.paper_id,
        authors=[],
        arxiv_id=workspace.paper_id,
        published_date="",
        abstract="",
        categories=[],
        citation_count=None,
    )


def _model_or_none(model_cls: type, payload: Any) -> Any | None:
    if payload is None:
        return None
    try:
        return model_cls.model_validate(payload)
    except Exception as exc:
        logger.warning("Skipping invalid %s payload: %s", model_cls.__name__, exc)
        return None


def _build_report_from_workspaces(
    workspaces: list[PaperWorkspace],
    *,
    claims: dict,
) -> ComparisonReport:
    papers = [
        _workspace_to_slot(workspace, index)
        for index, workspace in enumerate(workspaces)
    ]
    papers_summary = _build_papers_summary(papers)
    matrix = _build_comparison_matrix(papers)
    unique_tasks = _build_unique_tasks_per_paper(papers, matrix)
    unique_rows = _build_unique_rows_per_paper(papers, matrix)
    matrix_stats = _build_matrix_stats(papers, matrix)
    return _normalize_comparison_report(
        claims,
        papers,
        matrix,
        unique_tasks,
        unique_rows,
        papers_summary,
        matrix_stats,
        producer="comparison_analyst",
    )


def _build_evidence_from_workspaces(workspaces: list[PaperWorkspace]) -> str:
    papers = [
        _workspace_to_slot(workspace, index)
        for index, workspace in enumerate(workspaces)
    ]
    papers_summary = _build_papers_summary(papers)
    matrix = _build_comparison_matrix(papers)
    unique_tasks = _build_unique_tasks_per_paper(papers, matrix)
    unique_rows = _build_unique_rows_per_paper(papers, matrix)
    matrix_stats = _build_matrix_stats(papers, matrix)
    evidence = json.loads(
        _build_evidence_json(
            papers,
            matrix,
            unique_tasks,
            unique_rows,
            papers_summary,
            matrix_stats,
        )
    )
    evidence["paper_ids_by_index"] = {
        index: workspace.paper_id
        for index, workspace in enumerate(workspaces)
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def compare_workspaces(
    *,
    session_id: str,
    workspaces: list[PaperWorkspace],
    prompt: str | None = None,
    config: RunnableConfig | None = None,
) -> ComparisonAnalystResult:
    if len(workspaces) < 2:
        raise ValueError("Comparison Analyst requires at least two PaperWorkspace records.")

    policy = resolve_agent_policy("comparison_analyst", config)
    persistence = _agent_run_persistence(config)
    run = _start_run(session_id=session_id, workspaces=workspaces, config=config)
    evidence_json = _build_evidence_from_workspaces(workspaces)

    claims: dict = {}
    raw, llm_error = _call_llm(evidence_json, policy=policy, prompt=prompt)
    run.llm_call_count += 1

    if llm_error:
        logger.warning(
            "Comparison Analyst LLM unavailable, using deterministic fallback: %s",
            llm_error,
        )
        fallback_reason = llm_error
    else:
        parsed, parse_error = _parse_claims(raw or "")
        if parsed is not None and parse_error is None:
            claims = parsed
            fallback_reason = None
        else:
            logger.warning(
                "Comparison Analyst parse failed, using deterministic fallback: %s",
                parse_error,
            )
            fallback_reason = parse_error

    report = _build_report_from_workspaces(workspaces, claims=claims)
    markdown = _render_comparison_markdown(report)

    details = {
        "policy_applied": _policy_snapshot(policy),
        "paper_ids": [workspace.paper_id for workspace in workspaces],
        "matrix_rows": len(report.comparison_matrix),
        "producer": report.producer,
    }
    if fallback_reason:
        run.fallback(
            output_ref="comparison_report",
            details={**details, "fallback_reason": fallback_reason},
        )
    else:
        run.complete(output_ref="comparison_report", details=details)
    persistence.save(run)

    return ComparisonAnalystResult(report=report, markdown=markdown, agent_run=run)
