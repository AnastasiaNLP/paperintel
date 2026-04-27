import importlib
import sys
import types


def _load_graph_with_stubs():
    stub_modules = {
        "agents.benchmark": ("benchmark_analyst_agent", lambda state: {}),
        "agents.comparator": ("comparator_agent", lambda state: {}),
        "agents.extraction": ("extraction_agent", lambda state: {}),
        "agents.human_review": ("human_review_node", lambda state: {}),
        "agents.ingestion": ("ingestion_agent", lambda state: {}),
        "agents.paper_failure_finalize": (
            "paper_failure_finalize_node",
            lambda state: {},
        ),
        "agents.readiness": ("readiness_agent", lambda state: {}),
        "agents.report": ("report_agent", lambda state: {}),
        "agents.report_finalize": ("report_finalize_node", lambda state: {}),
    }

    for module_name, (attr_name, fn) in stub_modules.items():
        module = types.ModuleType(module_name)
        setattr(module, attr_name, fn)
        sys.modules[module_name] = module

    sys.modules.pop("graph", None)
    return importlib.import_module("graph")


def test_build_graph_compiles_with_batch_wiring():
    graph_module = _load_graph_with_stubs()
    graph = graph_module.build_graph()
    assert graph is not None


def test_create_app_without_checkpointing_compiles():
    graph_module = _load_graph_with_stubs()
    app = graph_module.create_app(use_checkpointing=False)
    assert app is not None
