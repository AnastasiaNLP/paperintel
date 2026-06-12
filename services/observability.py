from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from threading import Lock
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
    "trace_id",
}

METRIC_LABEL_FIELDS = {
    "agent_name",
    "failure_class",
    "kind",
    "model",
    "operation",
    "provider",
    "status",
    "termination_reason",
}

DURATION_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)

TRACE_ID_ENV_VARS = (
    "PAPERINTEL_TRACE_ID",
    "LANGSMITH_TRACE_ID",
    "LANGCHAIN_TRACE_ID",
    "LANGCHAIN_RUN_ID",
)


class ObservabilityMetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._event_counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._duration_counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._duration_sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(
            float
        )
        self._duration_buckets: dict[tuple[tuple[str, str], ...], dict[float, int]] = (
            defaultdict(lambda: defaultdict(int))
        )

    def record_event(self, event: str, fields: dict[str, Any]) -> None:
        labels = _metric_labels(event, fields)
        duration_ms = fields.get("duration_ms")
        with self._lock:
            self._event_counts[labels] += 1
            if isinstance(duration_ms, int | float):
                value = max(0.0, float(duration_ms))
                self._duration_counts[labels] += 1
                self._duration_sums[labels] += value
                for bucket in DURATION_BUCKETS_MS:
                    if value <= bucket:
                        self._duration_buckets[labels][float(bucket)] += 1
                self._duration_buckets[labels][float("inf")] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            event_counts = dict(self._event_counts)
            duration_counts = dict(self._duration_counts)
            duration_sums = dict(self._duration_sums)
            duration_buckets = {
                labels: dict(buckets)
                for labels, buckets in self._duration_buckets.items()
            }

        lines = [
            "# HELP paperintel_events_total Total safe observability events emitted.",
            "# TYPE paperintel_events_total counter",
        ]
        for labels, count in sorted(event_counts.items()):
            lines.append(
                f"paperintel_events_total{_format_metric_labels(labels)} {count}"
            )

        lines.extend(
            [
                "# HELP paperintel_event_duration_ms Safe event durations in milliseconds.",
                "# TYPE paperintel_event_duration_ms histogram",
            ]
        )
        for labels, buckets in sorted(duration_buckets.items()):
            for bucket in list(DURATION_BUCKETS_MS) + [float("inf")]:
                bucket_labels = tuple(labels) + (("le", _format_bucket(bucket)),)
                lines.append(
                    "paperintel_event_duration_ms_bucket"
                    f"{_format_metric_labels(bucket_labels)} "
                    f"{buckets.get(float(bucket), 0)}"
                )
            lines.append(
                "paperintel_event_duration_ms_count"
                f"{_format_metric_labels(labels)} {duration_counts.get(labels, 0)}"
            )
            lines.append(
                "paperintel_event_duration_ms_sum"
                f"{_format_metric_labels(labels)} {duration_sums.get(labels, 0.0)}"
            )
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._event_counts.clear()
            self._duration_counts.clear()
            self._duration_sums.clear()
            self._duration_buckets.clear()


_METRICS = ObservabilityMetricsRegistry()

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
    if "trace_id" not in safe_fields:
        trace_id = current_trace_id()
        if trace_id:
            safe_fields["trace_id"] = trace_id
    _METRICS.record_event(event, safe_fields)
    message = " ".join(
        ["event=" + event]
        + [f"{key}={_format_value(value)}" for key, value in safe_fields.items()]
    )
    logger.log(level, message)


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def render_prometheus_metrics() -> str:
    return _METRICS.render_prometheus()


def reset_observability_metrics() -> None:
    _METRICS.reset()


def current_trace_id() -> str | None:
    for key in TRACE_ID_ENV_VARS:
        value = _normalize_trace_id(os.getenv(key))
        if value:
            return value

    langsmith_trace_id = _current_langsmith_trace_id()
    if langsmith_trace_id:
        return langsmith_trace_id
    return None


def attach_observability_details(
    details: dict[str, Any],
    *,
    trace_id: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    observability = details.get("observability")
    if not isinstance(observability, dict):
        observability = {}
    existing_trace_id = _normalize_trace_id(observability.get("trace_id"))
    if existing_trace_id:
        observability["trace_id"] = existing_trace_id
    else:
        observability.pop("trace_id", None)
    resolved_trace_id = _normalize_trace_id(trace_id) or current_trace_id()
    if resolved_trace_id and "trace_id" not in observability:
        observability["trace_id"] = resolved_trace_id
    if duration_ms is not None:
        observability["duration_ms"] = max(0, int(duration_ms))
    if observability:
        details["observability"] = observability
    return details


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
        if key == "trace_id":
            value = _normalize_trace_id(value)
            if value is None:
                continue
        safe[key] = value
    return safe


def _normalize_trace_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if any(marker in normalized.lower() for marker in FORBIDDEN_FIELD_MARKERS):
        return None
    return normalized[:128]


def _current_langsmith_trace_id() -> str | None:
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return None
    try:
        run_tree = get_current_run_tree()
    except Exception:
        return None
    if run_tree is None:
        return None
    for attr in ("trace_id", "id", "run_id"):
        value = _normalize_trace_id(str(getattr(run_tree, attr, "") or ""))
        if value:
            return value
    return None


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _metric_labels(event: str, fields: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    labels = {"event": event}
    for key in sorted(METRIC_LABEL_FIELDS):
        value = fields.get(key)
        if value is None:
            continue
        labels[key] = str(value)
    return tuple(sorted(labels.items()))


def _format_metric_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(
        f'{key}="{_escape_metric_label(value)}"' for key, value in labels
    ) + "}"


def _escape_metric_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_bucket(bucket: float) -> str:
    if bucket == float("inf"):
        return "+Inf"
    return str(int(bucket))
