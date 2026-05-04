import importlib
import sys
import types

import pytest

from models.schemas import EngineerReport, PaperSlot


STUB_MODULE_NAMES = [
    "agents.benchmark",
    "agents.comparator",
    "agents.extraction",
    "agents.evidence_critic",
    "agents.human_review",
    "agents.ingestion",
    "agents.paper_failure_finalize",
    "agents.readiness",
    "agents.report",
    "agents.report_finalize",
]


@pytest.fixture(autouse=True)
def _cleanup_stubbed_graph_modules():
    original_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in STUB_MODULE_NAMES + ["graph"]
    }
    yield
    for module_name, original in original_modules.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _single_success_state() -> dict:
    return {
        "input_type": "url",
        "input_value": "https://arxiv.org/abs/2501.12948",
        "batch_urls": None,
        "papers": [],
        "metadata": None,
        "raw_text": None,
        "pdf_path": None,
        "text_by_page": None,
        "method_extraction": None,
        "benchmarks": [],
        "production_readiness": None,
        "ingestion_provenance": None,
        "comparison_markdown": None,
        "comparison_report": None,
        "engineer_report": None,
        "full_markdown_report": None,
        "current_paper_index": 0,
        "total_papers": 1,
        "processing_stage": "ingestion",
        "needs_human_review": False,
        "human_review_reason": None,
        "confidence_scores": {},
        "paper_failed": False,
        "paper_failure_reason": None,
        "failed_node": None,
        "messages": [],
        "errors": [],
        "agent_runs": [],
        "cost_tracking": {},
    }


def _batch_state(urls: list[str]) -> dict:
    state = _single_success_state()
    state["batch_urls"] = urls
    state["input_value"] = "ignored-in-batch"
    state["total_papers"] = len(urls)
    return state


class Scenario:
    def __init__(self, fail_indexes=None, fatal=False):
        self.fail_indexes = set(fail_indexes or [])
        self.fatal = fatal
        self.comparator_called = False

    def ingestion(self, state):
        if self.fatal:
            return {
                "processing_stage": "failed",
                "errors": ["fatal ingestion failure"],
            }

        current_index = state.get("current_paper_index", 0)
        if current_index in self.fail_indexes:
            return {
                "processing_stage": "paper_failure_finalize",
                "paper_failed": True,
                "paper_failure_reason": f"paper {current_index} failed",
                "failed_node": "ingestion",
                "errors": [f"paper {current_index} failed"],
            }

        return {
            "processing_stage": "extraction",
            "raw_text": f"paper {current_index} text",
            "errors": [],
        }

    def extraction(self, state):
        return {"processing_stage": "benchmark", "needs_human_review": False}

    def benchmark(self, state):
        return {"processing_stage": "readiness", "benchmarks": []}

    def readiness(self, state):
        return {"processing_stage": "report"}

    def report(self, state):
        current_index = state.get("current_paper_index", 0)
        return {
            "processing_stage": "completed",
            "engineer_report": EngineerReport(
                executive_summary=f"Paper {current_index} summary",
                key_innovation=f"Paper {current_index} innovation",
                practical_implications=f"Paper {current_index} implications",
                implementation_difficulty="moderate",
                recommended_action="prototype",
                action_reasoning=f"Paper {current_index} reasoning",
            ),
            "full_markdown_report": f"# Paper {current_index}",
        }

    def evidence_critic(self, state):
        return {}

    def report_finalize(self, state):
        current_index = state.get("current_paper_index", 0)
        input_url = (
            state["batch_urls"][current_index]
            if state.get("batch_urls")
            else state.get("input_value", "")
        )
        slot = PaperSlot(
            paper_index=current_index,
            input_url=input_url,
            engineer_report=state.get("engineer_report"),
            markdown_report=state.get("full_markdown_report"),
            errors=list(state.get("errors", []) or []),
            completed=True,
        )
        return {
            "papers": [slot],
            "current_paper_index": current_index + 1,
            "processing_stage": "report_finalize",
            "errors": [],
            "metadata": None,
            "raw_text": None,
            "pdf_path": None,
            "text_by_page": None,
            "method_extraction": None,
            "benchmarks": [],
            "production_readiness": None,
            "engineer_report": None,
            "full_markdown_report": None,
            "ingestion_provenance": None,
            "confidence_scores": {},
            "needs_human_review": False,
            "human_review_reason": None,
            "paper_failed": False,
            "paper_failure_reason": None,
            "failed_node": None,
        }

    def paper_failure_finalize(self, state):
        current_index = state.get("current_paper_index", 0)
        input_url = (
            state["batch_urls"][current_index]
            if state.get("batch_urls")
            else state.get("input_value", "")
        )
        slot = PaperSlot(
            paper_index=current_index,
            input_url=input_url,
            errors=list(state.get("errors", []) or []),
            completed=False,
        )
        return {
            "papers": [slot],
            "current_paper_index": current_index + 1,
            "processing_stage": "paper_failure_finalize",
            "errors": [],
            "metadata": None,
            "raw_text": None,
            "pdf_path": None,
            "text_by_page": None,
            "method_extraction": None,
            "benchmarks": [],
            "production_readiness": None,
            "engineer_report": None,
            "full_markdown_report": None,
            "ingestion_provenance": None,
            "confidence_scores": {},
            "needs_human_review": False,
            "human_review_reason": None,
            "paper_failed": False,
            "paper_failure_reason": None,
            "failed_node": None,
        }

    def comparator(self, state):
        self.comparator_called = True
        return {
            "processing_stage": "comparison_completed",
            "comparison_report": {
                "paper_count": len(state.get("papers", [])),
                "completed_count": sum(
                    1 for paper in state.get("papers", []) if getattr(paper, "completed", False)
                ),
            },
            "comparison_markdown": "# Comparison",
        }


def _load_graph_with_scenario(scenario: Scenario):
    stub_modules = {
        "agents.benchmark": ("benchmark_analyst_agent", scenario.benchmark),
        "agents.comparator": ("comparator_agent", scenario.comparator),
        "agents.extraction": ("extraction_agent", scenario.extraction),
        "agents.evidence_critic": ("evidence_critic_agent", scenario.evidence_critic),
        "agents.human_review": ("human_review_node", lambda state: {}),
        "agents.ingestion": ("ingestion_agent", scenario.ingestion),
        "agents.paper_failure_finalize": (
            "paper_failure_finalize_node",
            scenario.paper_failure_finalize,
        ),
        "agents.readiness": ("readiness_agent", scenario.readiness),
        "agents.report": ("report_agent", scenario.report),
        "agents.report_finalize": ("report_finalize_node", scenario.report_finalize),
    }

    for module_name, (attr_name, fn) in stub_modules.items():
        module = types.ModuleType(module_name)
        setattr(module, attr_name, fn)
        sys.modules[module_name] = module

    sys.modules.pop("graph", None)
    return importlib.import_module("graph")


def test_single_success_flow_creates_one_completed_slot_and_ends():
    scenario = Scenario()
    graph_module = _load_graph_with_scenario(scenario)
    app = graph_module.create_app(use_checkpointing=False)

    result = app.invoke(_single_success_state())

    assert result["current_paper_index"] == 1
    assert len(result["papers"]) == 1
    assert result["papers"][0].completed is True
    assert scenario.comparator_called is False


def test_batch_success_flow_runs_comparator_after_all_papers():
    scenario = Scenario()
    graph_module = _load_graph_with_scenario(scenario)
    app = graph_module.create_app(use_checkpointing=False)

    result = app.invoke(
        _batch_state(
            [
                "https://arxiv.org/abs/2501.12948",
                "https://arxiv.org/abs/2305.14314",
            ]
        )
    )

    assert result["processing_stage"] == "comparison_completed"
    assert len(result["papers"]) == 2
    assert all(paper.completed is True for paper in result["papers"])
    assert scenario.comparator_called is True
    assert result["comparison_report"]["paper_count"] == 2


def test_batch_flow_continues_after_one_failed_paper():
    scenario = Scenario(fail_indexes={1})
    graph_module = _load_graph_with_scenario(scenario)
    app = graph_module.create_app(use_checkpointing=False)

    result = app.invoke(
        _batch_state(
            [
                "https://arxiv.org/abs/2501.12948",
                "https://arxiv.org/abs/2305.14314",
                "https://arxiv.org/abs/2310.06825",
            ]
        )
    )

    assert result["processing_stage"] == "comparison_completed"
    assert len(result["papers"]) == 3
    assert result["papers"][0].completed is True
    assert result["papers"][1].completed is False
    assert result["papers"][2].completed is True
    assert scenario.comparator_called is True
    assert result["comparison_report"]["completed_count"] == 2


def test_single_fatal_failure_ends_without_finalize_or_comparator():
    scenario = Scenario(fatal=True)
    graph_module = _load_graph_with_scenario(scenario)
    app = graph_module.create_app(use_checkpointing=False)

    result = app.invoke(_single_success_state())

    assert result["processing_stage"] == "failed"
    assert result.get("papers", []) == []
    assert scenario.comparator_called is False
