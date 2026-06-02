from collections.abc import Callable
from inspect import signature
from typing import Any

from langchain_core.runnables import RunnableConfig


class WorkflowCancellationRequested(RuntimeError):
    pass


def check_cancellation(config: RunnableConfig | None) -> None:
    if not isinstance(config, dict):
        return
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return
    callback = configurable.get("cancellation_callback")
    if isinstance(callback, Callable):
        callback()


def cancellation_guard(node: Callable) -> Callable:
    accepts_config = len(signature(node).parameters) >= 2

    def guarded(state: dict[str, Any], config: RunnableConfig | None = None):
        check_cancellation(config)
        if accepts_config:
            return node(state, config)
        return node(state)

    return guarded
