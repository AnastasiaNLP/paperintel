import pytest

from agents.cancellation import cancellation_guard


def test_cancellation_guard_checks_callback_before_node_execution():
    calls = []

    def cancel():
        calls.append("cancel")
        raise RuntimeError("stop")

    def node(state):
        calls.append("node")
        return state

    with pytest.raises(RuntimeError, match="stop"):
        cancellation_guard(node)(
            {"processing_stage": "ingestion"},
            {"configurable": {"cancellation_callback": cancel}},
        )

    assert calls == ["cancel"]


def test_cancellation_guard_passes_config_to_config_aware_node():
    config = {"configurable": {}}

    def node(state, received_config):
        return {"state": state, "config": received_config}

    result = cancellation_guard(node)({"ok": True}, config)

    assert result == {"state": {"ok": True}, "config": config}
