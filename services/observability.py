from __future__ import annotations

import json
import logging
import time
from typing import Any


SAFE_EVENT_FIELDS = {
    "session_id",
    "turn_id",
    "job_id",
    "agent_run_id",
    "agent_name",
    "paper_id",
    "provider",
    "operation",
    "model",
    "failure_class",
    "retryable",
    "retry_after_seconds",
    "termination_reason",
    "attempts",
    "max_attempts",
    "duration_ms",
    "timeout_seconds",
    "status",
    "kind",
    "worker_id",
    "result_size",
    "result_count",
}

FORBIDDEN_FIELD_MARKERS = {
    "api_key",
    "password",
    "secret",
    "token",
    "system_prompt",
    "user_content",
    "raw_text",
    "pdf_text",
    "traceback",
}


def emit_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    safe_fields = _safe_event_fields(fields)
    message = " ".join(
        ["event=" + event]
        + [f"{key}={_format_value(value)}" for key, value in safe_fields.items()]
    )
    logger.log(level, message)


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _safe_event_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in fields.items():
        normalized = key.lower()
        if any(marker in normalized for marker in FORBIDDEN_FIELD_MARKERS):
            continue
        if key not in SAFE_EVENT_FIELDS:
            continue
        if value is None:
            continue
        safe[key] = value
    return safe


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
