"""
Live sanity test for agents.comparator.comparator_agent on a comparable pair.

Pair: DeepSeek-R1 vs DeepSeek-V3.

This test DOES call Anthropic for the Comparator step.
It does NOT call LangGraph, arXiv, Semantic Scholar, GitHub, HuggingFace, or PDF parsing.

Run:
  python test_comparator_live_deepseek.py
"""

import logging
import sys

from agents.comparator import comparator_agent
from config.settings import settings
from models.schemas import (
    BenchmarkResult,
    EngineerReport,
    MethodExtraction,
    PaperMetadata,
    PaperSlot,
    ProductionReadiness,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


def _deepseek_r1_slot() -> PaperSlot:
    return PaperSlot(
        paper_index=0,
        input_url="https://arxiv.org/abs/2501.12948",
        metadata=PaperMetadata(
            title="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
            authors=["DeepSeek-AI"],
            arxiv_id="2501.12948",
            published_date="2025-01-22",
            abstract=(
                "DeepSeek-R1 explores reinforcement learning for reasoning in language models "
                "and releases open reasoning models and distilled variants."
            ),
            categories=["cs.CL", "cs.AI"],
            citation_count=5350,
        ),
        method_extraction=MethodExtraction(
            method_name="DeepSeek-R1",
            description=(
                "Reasoning-focused language model trained with large-scale reinforcement learning, "
                "GRPO-style optimization, rule-based rewards, and distillation."
            ),
            novelty_claim=(
                "Strong reasoning behavior can emerge from reinforcement learning post-training, "
                "including a pure-RL path before supervised fine-tuning."
            ),
            key_components=["GRPO", "rule-based rewards", "cold-start data", "distillation"],
            compared_to=["DeepSeek-V3", "OpenAI o1", "GPT-4o", "Claude"],
            limitations_stated=[
                "Long-context and multi-turn behavior require further improvement.",
                "Distilled models trade accuracy for deployment cost.",
            ],
        ),
        benchmarks=[
            BenchmarkResult(task="MATH-500", metric="pass@1", value=97.3, conditions="reasoning evaluation"),
            BenchmarkResult(task="AIME 2024", metric="pass@1", value=79.8, conditions="math reasoning"),
            BenchmarkResult(task="GPQA Diamond", metric="accuracy", value=71.5, unit="%", conditions="science reasoning"),
            BenchmarkResult(task="HumanEval", metric="pass@1", value=65.9, conditions="code generation"),
            BenchmarkResult(task="MMLU", metric="accuracy", value=90.8, unit="%", conditions="general knowledge"),
        ],
        production_readiness=ProductionReadiness(
            has_open_code=True,
            code_url="https://github.com/deepseek-ai/DeepSeek-R1",
            huggingface_model="deepseek-ai/DeepSeek-R1",
            framework_integrations=["HuggingFace Transformers", "vLLM"],
            min_gpu_requirement=None,
            estimated_inference_cost=None,
            dependencies=[],
            maturity_level="experimental",
            maturity_reasoning=(
                "Verified open code and HuggingFace model are available. Maturity remains "
                "experimental because deployment cost and production validation are not established."
            ),
        ),
        engineer_report=EngineerReport(
            executive_summary="DeepSeek-R1 is a strong open reasoning model worth prototyping.",
            key_innovation="Reinforcement learning post-training for reasoning behavior.",
            practical_implications=(
                "Good fit for reasoning-heavy workloads, with deployment validation still required."
            ),
            implementation_difficulty="moderate",
            recommended_action="prototype",
            action_reasoning=(
                "Experimental maturity with verified model/code availability and strong reasoning benchmarks."
            ),
        ),
        markdown_report="# DeepSeek-R1",
        completed=True,
        errors=[],
    )


def _deepseek_v3_slot() -> PaperSlot:
    return PaperSlot(
        paper_index=1,
        input_url="https://arxiv.org/abs/2412.19437",
        metadata=PaperMetadata(
            title="DeepSeek-V3 Technical Report",
            authors=["DeepSeek-AI"],
            arxiv_id="2412.19437",
            published_date="2024-12-27",
            abstract=(
                "DeepSeek-V3 is a strong mixture-of-experts language model optimized for general "
                "chat, coding, math, and knowledge tasks."
            ),
            categories=["cs.CL", "cs.AI"],
            citation_count=1200,
        ),
        method_extraction=MethodExtraction(
            method_name="DeepSeek-V3",
            description=(
                "Large-scale mixture-of-experts language model with efficient training and strong "
                "general-purpose benchmark performance."
            ),
            novelty_claim=(
                "Efficient MoE scaling and training techniques deliver strong performance across "
                "general language, coding, and math benchmarks."
            ),
            key_components=["mixture-of-experts", "multi-token prediction", "efficient training"],
            compared_to=["DeepSeek-V2", "GPT-4o", "Claude", "Llama"],
            limitations_stated=[
                "Not specialized for long chain-of-thought reasoning to the same degree as R1.",
                "Large model serving still requires serious deployment validation.",
            ],
        ),
        benchmarks=[
            BenchmarkResult(task="MATH-500", metric="pass@1", value=90.2, conditions="reasoning evaluation"),
            BenchmarkResult(task="AIME 2024", metric="pass@1", value=39.2, conditions="math reasoning"),
            BenchmarkResult(task="GPQA Diamond", metric="accuracy", value=59.1, unit="%", conditions="science reasoning"),
            BenchmarkResult(task="HumanEval", metric="pass@1", value=82.6, conditions="code generation"),
            BenchmarkResult(task="MMLU", metric="accuracy", value=88.5, unit="%", conditions="general knowledge"),
        ],
        production_readiness=ProductionReadiness(
            has_open_code=True,
            code_url=None,
            huggingface_model="deepseek-ai/DeepSeek-V3",
            framework_integrations=["HuggingFace Transformers", "vLLM"],
            min_gpu_requirement=None,
            estimated_inference_cost=None,
            dependencies=[],
            maturity_level="experimental",
            maturity_reasoning=(
                "Verified HuggingFace model availability and standard serving integrations exist, "
                "but production cost and deployment validation remain open."
            ),
        ),
        engineer_report=EngineerReport(
            executive_summary=(
                "DeepSeek-V3 is a strong general-purpose open model suited for chat, coding, "
                "and broad language tasks."
            ),
            key_innovation="Efficient MoE scaling for broad benchmark performance.",
            practical_implications=(
                "Good fit for general assistant and coding workloads where broad capability matters "
                "more than specialized reasoning."
            ),
            implementation_difficulty="moderate",
            recommended_action="prototype",
            action_reasoning=(
                "Experimental maturity with verified model availability and strong general benchmarks."
            ),
        ),
        markdown_report="# DeepSeek-V3",
        completed=True,
        errors=[],
    )


def _assert_valid_result(result: dict) -> None:
    assert result["processing_stage"] == "comparison_completed"

    report = result.get("comparison_report")
    markdown = result.get("comparison_markdown")

    assert report is not None, "comparison_report missing"
    assert markdown and isinstance(markdown, str), "comparison_markdown missing"
    assert len(report.papers_summary) == 2
    assert report.trade_offs.strip(), "trade_offs is empty"
    assert report.overall_winner_reasoning.strip(), "overall_winner_reasoning is empty"
    assert report.winner_basis in {
        "readiness_dominant",
        "benchmark_dominant",
        "mixed",
        "no_clear_winner",
    }
    assert report.rows_with_winner >= 3
    assert report.benchmark_overlap_ratio > 0.5

    valid_indexes = {0, 1}
    if report.overall_winner_index is not None:
        assert report.overall_winner_index in valid_indexes

    rows_with_both_values = [
        row
        for row in report.comparison_matrix
        if row.values_by_paper.get(0) is not None and row.values_by_paper.get(1) is not None
    ]
    rows_with_winner = [row for row in report.comparison_matrix if row.winner_index is not None]

    assert len(rows_with_both_values) >= 4, "expected several aligned benchmark rows"
    assert len(rows_with_winner) >= 3, "expected winners on comparable benchmark rows"

    for rec in report.recommendations:
        assert rec.recommended_paper_index in valid_indexes
        assert rec.constraint.strip()
        assert rec.reasoning.strip()

    assert "# Paper Comparison" in markdown
    assert "## Benchmark Matrix" in markdown
    assert "Paper 0" in markdown
    assert "Paper 1" in markdown
    assert "## Recommendations" in markdown
    assert "## Overall" in markdown

    dumped = report.model_dump()
    assert dumped["comparison_matrix"]


def main() -> int:
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not configured", file=sys.stderr)
        return 2

    state = {
        "papers": [_deepseek_r1_slot(), _deepseek_v3_slot()],
        "processing_stage": "comparator",
        "errors": [],
    }

    print("=== Live comparator sanity: DeepSeek-R1 vs DeepSeek-V3 ===")
    print("Calls Anthropic only for the Comparator step.")
    print()

    result = comparator_agent(state)
    _assert_valid_result(result)

    report = result["comparison_report"]
    markdown = result["comparison_markdown"]

    print("stage:", result["processing_stage"])
    print("overall_winner_index:", report.overall_winner_index)
    print("winner_basis:", report.winner_basis)
    print("matrix_rows:", len(report.comparison_matrix))
    print("rows_with_winner:", len([row for row in report.comparison_matrix if row.winner_index is not None]))
    print("benchmark_overlap_ratio:", report.benchmark_overlap_ratio)
    print()
    print("=== Trade-offs ===")
    print(report.trade_offs)
    print()
    print("=== Recommendations ===")
    if not report.recommendations:
        print("_none_")
    else:
        for rec in report.recommendations:
            print(f"- {rec.constraint}: Paper {rec.recommended_paper_index}")
            print(f"  {rec.reasoning}")
    print()
    print("=== Overall ===")
    print(report.overall_winner_reasoning)
    print()
    print("=== Markdown ===")
    print(markdown)
    print()
    print("Live comparable comparator sanity: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
