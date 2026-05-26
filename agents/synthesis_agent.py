from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agents.agent_run_recorder import AgentRunPersistence, NoopAgentRunPersistence
from agents.comparison_analyst import _model_or_none
from agents.llm_provider import call_text_llm
from config.settings import settings
from models.agent_policies import AgentRuntimePolicy, resolve_agent_policy
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.schemas import (
    BenchmarkResult,
    EngineerReport,
    MethodExtraction,
    ProductionReadiness,
)
from models.session import Persona
from models.synthesis import (
    SynthesisAgentResult,
    SynthesisCitation,
    SynthesisRecommendation,
    SynthesisReport,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the PaperIntel Synthesis Agent.

Return ONLY a JSON object. No markdown, no prose.

Use only the durable PaperWorkspace evidence and optional persisted comparison
context. Do not invent paper ids, benchmark values, code availability, or
implementation claims. Tailor the answer to the requested persona:
- engineer: implementation, dependencies, risks, next engineering step.
- researcher: novelty, baselines, evidence quality, open questions.
- techlead: ROI, maturity, adoption risk, sequencing.

Expected JSON shape:
{
  "persona": "engineer | researcher | techlead",
  "summary": "string",
  "key_takeaways": ["string"],
  "trade_offs": ["string"],
  "recommended_next_steps": [
    {"recommendation": "string", "reasoning": "string"}
  ],
  "citations": [
    {"paper_id": "string", "quote_or_summary": "string"}
  ],
  "limitations": ["string"]
}
"""


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
    comparison: ComparisonArtifact | None,
    config: RunnableConfig | None,
) -> AgentRun:
    configurable = _configurable(config)
    input_refs = [f"paper_workspace:{workspace.paper_id}" for workspace in workspaces]
    if comparison is not None:
        input_refs.append(f"comparison_artifact:{comparison.id}")
    return AgentRun(
        agent_name="synthesis_agent",
        session_id=session_id,
        job_id=configurable.get("job_id"),
        input_refs=input_refs,
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
    Single-pass synthesis call.

    CA.2 intentionally skips a JSON repair call. Invalid or unavailable LLM
    output falls back to the deterministic durable-artifact summary, keeping the
    service-level synthesis path bounded to one LLM call.
    """
    user_content = evidence_json
    if prompt and prompt.strip():
        user_content += "\n\nUser synthesis prompt:\n" + prompt.strip()
    return call_text_llm(
        requested_model=settings.sonnet_model,
        system_prompt=_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=policy.max_tokens or 4_000,
        context_label="Synthesis Agent LLM",
    )


def _build_evidence(
    *,
    persona: Persona,
    workspaces: list[PaperWorkspace],
    comparison: ComparisonArtifact | None,
) -> str:
    evidence: dict[str, Any] = {
        "persona": persona,
        "papers": [_workspace_evidence(workspace) for workspace in workspaces],
    }
    if comparison is not None:
        evidence["latest_comparison"] = {
            "id": comparison.id,
            "paper_ids": comparison.paper_ids,
            "comparison_report": comparison.comparison_report_json,
            "comparison_markdown": comparison.comparison_markdown,
        }
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _workspace_evidence(workspace: PaperWorkspace) -> dict[str, Any]:
    method = _model_or_none(MethodExtraction, workspace.method_extraction_json)
    readiness = _model_or_none(ProductionReadiness, workspace.readiness_json)
    report = _model_or_none(EngineerReport, workspace.finalized_report_json)
    benchmarks = [
        benchmark.model_dump(mode="json")
        for item in workspace.benchmarks_json
        if (benchmark := _model_or_none(BenchmarkResult, item)) is not None
    ]
    evidence = {
        "paper_id": workspace.paper_id,
        "title": workspace.title or workspace.paper_id,
        "source_url": workspace.source_url,
        "pipeline_stage": workspace.pipeline_stage,
        "method_extraction": method.model_dump(mode="json") if method else None,
        "benchmarks": benchmarks,
        "production_readiness": readiness.model_dump(mode="json") if readiness else None,
        "engineer_report": report.model_dump(mode="json") if report else None,
        "finalized_report_markdown": workspace.full_markdown_report,
    }
    evidence["limitations"] = _workspace_limitations(workspace, method, readiness, report)
    return evidence


def _workspace_limitations(
    workspace: PaperWorkspace,
    method: MethodExtraction | None,
    readiness: ProductionReadiness | None,
    report: EngineerReport | None,
) -> list[str]:
    limitations = []
    if method is None:
        limitations.append("method extraction missing")
    elif method.limitations_stated:
        limitations.extend(method.limitations_stated)
    if not workspace.benchmarks_json:
        limitations.append("benchmark rows missing")
    if readiness is None:
        limitations.append("readiness extraction missing")
    if report is None:
        limitations.append("engineer report missing")
    return limitations


def _parse_report(
    raw: str,
    *,
    persona: Persona,
    paper_ids: set[str],
) -> tuple[SynthesisReport | None, str | None]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload["persona"] = persona
            payload["citations"] = [
                citation
                for citation in payload.get("citations", [])
                if isinstance(citation, dict) and citation.get("paper_id") in paper_ids
            ]
        report = SynthesisReport.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        return None, str(exc)
    if not report.citations:
        return None, "LLM synthesis did not cite selected paper ids."
    return report, None


def _fallback_report(
    *,
    persona: Persona,
    workspaces: list[PaperWorkspace],
    comparison: ComparisonArtifact | None,
) -> SynthesisReport:
    paper_labels = [
        f"{workspace.paper_id} ({workspace.title or workspace.paper_id})"
        for workspace in workspaces
    ]
    summary = _persona_summary(persona, paper_labels)
    takeaways = [_paper_takeaway(workspace, persona) for workspace in workspaces]
    trade_offs = _trade_offs(workspaces, comparison)
    citations = [
        SynthesisCitation(
            paper_id=workspace.paper_id,
            quote_or_summary=_citation_summary(workspace),
        )
        for workspace in workspaces
    ]
    return SynthesisReport(
        persona=persona,
        summary=summary,
        key_takeaways=takeaways,
        trade_offs=trade_offs,
        recommended_next_steps=_next_steps(persona, workspaces),
        citations=citations,
        limitations=_limitations(workspaces),
    )


def _persona_summary(persona: Persona, paper_labels: list[str]) -> str:
    papers = ", ".join(paper_labels)
    if persona == "researcher":
        return f"Research synthesis across {papers}, focused on novelty and evidence quality."
    if persona == "techlead":
        return f"Technical leadership synthesis across {papers}, focused on maturity and adoption risk."
    return f"Engineering synthesis across {papers}, focused on implementation trade-offs."


def _paper_takeaway(workspace: PaperWorkspace, persona: Persona) -> str:
    method = _model_or_none(MethodExtraction, workspace.method_extraction_json)
    readiness = _model_or_none(ProductionReadiness, workspace.readiness_json)
    report = _model_or_none(EngineerReport, workspace.finalized_report_json)
    method_name = method.method_name if method else workspace.title or workspace.paper_id
    if persona == "researcher":
        novelty = method.novelty_claim if method else "novelty evidence is missing"
        return f"{workspace.paper_id}: {method_name} - {novelty}."
    if persona == "techlead":
        maturity = readiness.maturity_level if readiness else "unknown maturity"
        action = report.recommended_action if report else "no action captured"
        return f"{workspace.paper_id}: {method_name} has {maturity}; recommended action is {action}."
    difficulty = report.implementation_difficulty if report else "unknown difficulty"
    deps = ", ".join(readiness.dependencies) if readiness and readiness.dependencies else "dependencies not captured"
    return f"{workspace.paper_id}: {method_name} looks {difficulty} to implement; {deps}."


def _trade_offs(
    workspaces: list[PaperWorkspace],
    comparison: ComparisonArtifact | None,
) -> list[str]:
    trade_offs = []
    if comparison is not None and comparison.comparison_markdown.strip():
        trade_offs.append("Latest persisted comparison is available and was used as context.")
    benchmark_counts = {
        workspace.paper_id: len(workspace.benchmarks_json)
        for workspace in workspaces
    }
    trade_offs.append(f"Benchmark evidence varies by paper: {benchmark_counts}.")
    readiness = []
    for workspace in workspaces:
        readiness_model = _model_or_none(ProductionReadiness, workspace.readiness_json)
        readiness.append(
            f"{workspace.paper_id}: {readiness_model.maturity_level if readiness_model else 'unknown'}"
        )
    trade_offs.append("Production maturity: " + "; ".join(readiness) + ".")
    return trade_offs


def _next_steps(
    persona: Persona,
    workspaces: list[PaperWorkspace],
) -> list[SynthesisRecommendation]:
    if persona == "researcher":
        return [
            SynthesisRecommendation(
                recommendation="Verify novelty against the cited baselines before adopting claims.",
                reasoning="The durable workspace can summarize extracted claims, but it cannot prove novelty without broader literature review.",
            )
        ]
    if persona == "techlead":
        return [
            SynthesisRecommendation(
                recommendation="Sequence a small technical spike before committing roadmap capacity.",
                reasoning="Readiness and benchmark evidence are uneven across the selected papers.",
            )
        ]
    return [
        SynthesisRecommendation(
            recommendation="Prototype the most implementation-ready method behind a feature flag.",
            reasoning="The selected workspaces expose enough method and readiness evidence for a bounded engineering trial.",
        )
    ]


def _citation_summary(workspace: PaperWorkspace) -> str:
    report = _model_or_none(EngineerReport, workspace.finalized_report_json)
    if report is not None:
        return report.executive_summary
    method = _model_or_none(MethodExtraction, workspace.method_extraction_json)
    if method is not None:
        return method.description
    return workspace.full_markdown_report or workspace.title or workspace.paper_id


def _limitations(workspaces: list[PaperWorkspace]) -> list[str]:
    limitations: list[str] = []
    for workspace in workspaces:
        workspace_limitations = _workspace_limitations(
            workspace,
            _model_or_none(MethodExtraction, workspace.method_extraction_json),
            _model_or_none(ProductionReadiness, workspace.readiness_json),
            _model_or_none(EngineerReport, workspace.finalized_report_json),
        )
        for limitation in workspace_limitations:
            limitations.append(f"{workspace.paper_id}: {limitation}")
    return limitations


def _render_response(report: SynthesisReport) -> str:
    lines = [
        f"Synthesis for {report.persona}",
        "",
        report.summary,
        "",
        "Key takeaways:",
    ]
    lines.extend(f"- {item}" for item in report.key_takeaways)
    lines.extend(["", "Trade-offs:"])
    lines.extend(f"- {item}" for item in report.trade_offs)
    lines.extend(["", "Recommended next steps:"])
    lines.extend(
        f"- {step.recommendation} {step.reasoning}"
        for step in report.recommended_next_steps
    )
    if report.limitations:
        lines.extend(["", "Limitations:"])
        lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines).strip()


def synthesize_workspaces(
    *,
    session_id: str,
    persona: Persona,
    workspaces: list[PaperWorkspace],
    prompt: str | None = None,
    comparison: ComparisonArtifact | None = None,
    config: RunnableConfig | None = None,
) -> SynthesisAgentResult:
    if len(workspaces) < 2:
        raise ValueError("Synthesis Agent requires at least two PaperWorkspace records.")

    policy = resolve_agent_policy("synthesis_agent", config)
    persistence = _agent_run_persistence(config)
    run = _start_run(
        session_id=session_id,
        workspaces=workspaces,
        comparison=comparison,
        config=config,
    )
    paper_ids = {workspace.paper_id for workspace in workspaces}
    evidence_json = _build_evidence(
        persona=persona,
        workspaces=workspaces,
        comparison=comparison,
    )

    fallback_reason = None
    raw, llm_error = _call_llm(evidence_json, policy=policy, prompt=prompt)
    run.llm_call_count += 1
    if llm_error:
        logger.warning("Synthesis Agent LLM unavailable, using fallback: %s", llm_error)
        fallback_reason = llm_error
        report = _fallback_report(
            persona=persona,
            workspaces=workspaces,
            comparison=comparison,
        )
    else:
        parsed, parse_error = _parse_report(raw or "", persona=persona, paper_ids=paper_ids)
        if parsed is None:
            logger.warning("Synthesis Agent parse failed, using fallback: %s", parse_error)
            fallback_reason = parse_error
            report = _fallback_report(
                persona=persona,
                workspaces=workspaces,
                comparison=comparison,
            )
        else:
            report = parsed

    response_text = _render_response(report)
    details = {
        "policy_applied": _policy_snapshot(policy),
        "paper_ids": [workspace.paper_id for workspace in workspaces],
        "persona": persona,
        "comparison_artifact_id": comparison.id if comparison is not None else None,
    }
    if fallback_reason:
        run.fallback(
            output_ref="synthesis_report",
            details={**details, "fallback_reason": fallback_reason},
        )
    else:
        run.complete(output_ref="synthesis_report", details=details)
    persistence.save(run)
    return SynthesisAgentResult(
        report=report,
        response_text=response_text,
        agent_run=run,
    )
