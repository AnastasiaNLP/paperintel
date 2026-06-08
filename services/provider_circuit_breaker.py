from __future__ import annotations

import logging
from typing import Protocol

from tools.circuit_breaker import CircuitBreakerOpenError


LOGGER = logging.getLogger(__name__)


class ProviderCircuitBreakerRepository(Protocol):
    def before_request(
        self,
        *,
        provider: str,
        operation: str,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ):
        ...

    def record_success(
        self,
        *,
        provider: str,
        operation: str,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        ...

    def record_failure(
        self,
        *,
        provider: str,
        operation: str,
        failure_threshold: int,
        recovery_timeout_seconds: float,
        failure_class: str | None = None,
    ) -> None:
        ...


class ProviderCircuitBreaker(Protocol):
    def before_request(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        ...

    def record_success(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        ...

    def record_failure(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
        failure_class: str | None = None,
    ) -> None:
        ...


class NoopProviderCircuitBreaker:
    def before_request(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        return None

    def record_success(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        return None

    def record_failure(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
        failure_class: str | None = None,
    ) -> None:
        return None


class PostgresProviderCircuitBreaker:
    def __init__(
        self,
        repository: ProviderCircuitBreakerRepository,
        *,
        fail_open: bool = True,
    ) -> None:
        self.repository = repository
        self.fail_open = fail_open

    def before_request(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        try:
            decision = self.repository.before_request(
                provider=provider,
                operation=operation,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )
        except Exception as exc:
            self._handle_repository_error("before_request", provider, operation, exc)
            return None
        if not decision.allowed:
            raise CircuitBreakerOpenError(provider, decision.retry_after_seconds)
        return None

    def record_success(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        try:
            self.repository.record_success(
                provider=provider,
                operation=operation,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )
        except Exception as exc:
            self._handle_repository_error("record_success", provider, operation, exc)

    def record_failure(
        self,
        provider: str,
        operation: str,
        *,
        failure_threshold: int,
        recovery_timeout_seconds: float,
        failure_class: str | None = None,
    ) -> None:
        try:
            self.repository.record_failure(
                provider=provider,
                operation=operation,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
                failure_class=failure_class,
            )
        except Exception as exc:
            self._handle_repository_error("record_failure", provider, operation, exc)

    def _handle_repository_error(
        self,
        action: str,
        provider: str,
        operation: str,
        exc: Exception,
    ) -> None:
        if not self.fail_open:
            raise exc
        LOGGER.warning(
            "Provider circuit breaker state unavailable during %s for %s/%s; continuing.",
            action,
            provider,
            operation,
            exc_info=exc,
        )
