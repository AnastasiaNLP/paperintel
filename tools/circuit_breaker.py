from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreakerOpenError(RuntimeError):
    def __init__(self, service_name: str, retry_after_seconds: float) -> None:
        super().__init__(
            f"{service_name} circuit breaker is open; retry after "
            f"{retry_after_seconds:.1f}s"
        )
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds


@dataclass
class CircuitBreaker:
    service_name: str
    failure_threshold: int
    recovery_timeout_seconds: float
    _state: BreakerState = "closed"
    _failure_count: int = 0
    _opened_at: float | None = None
    _lock: Lock = field(default_factory=Lock)

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._transition_if_recoverable(time.monotonic())
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def before_request(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._transition_if_recoverable(now)
            if self._state == "open":
                retry_after = self.recovery_timeout_seconds
                if self._opened_at is not None:
                    retry_after = max(
                        0.0,
                        self.recovery_timeout_seconds - (now - self._opened_at),
                    )
                raise CircuitBreakerOpenError(self.service_name, retry_after)

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = None

    def _transition_if_recoverable(self, now: float) -> None:
        if self._state != "open" or self._opened_at is None:
            return
        if now - self._opened_at >= self.recovery_timeout_seconds:
            self._state = "half_open"
