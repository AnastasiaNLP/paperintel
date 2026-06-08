from __future__ import annotations

import logging
import time
from typing import Protocol


LOGGER = logging.getLogger(__name__)


class ProviderRateLimitRepository(Protocol):
    def reserve_slot(
        self,
        *,
        provider: str,
        operation: str,
        interval_seconds: float,
    ):
        ...


class ProviderRateLimiter(Protocol):
    def acquire(
        self,
        provider: str,
        operation: str,
        *,
        interval_seconds: float,
    ) -> None:
        ...


class NoopProviderRateLimiter:
    def acquire(
        self,
        provider: str,
        operation: str,
        *,
        interval_seconds: float,
    ) -> None:
        return None


class PostgresProviderRateLimiter:
    def __init__(
        self,
        repository: ProviderRateLimitRepository,
        *,
        fail_open: bool = True,
        fallback_to_local: bool = True,
        sleeper=time.sleep,
    ) -> None:
        self.repository = repository
        self.fail_open = fail_open
        self.fallback_to_local = fallback_to_local
        self.sleeper = sleeper

    def acquire(
        self,
        provider: str,
        operation: str,
        *,
        interval_seconds: float,
    ) -> None:
        try:
            reservation = self.repository.reserve_slot(
                provider=provider,
                operation=operation,
                interval_seconds=interval_seconds,
            )
        except Exception as exc:
            if not self.fail_open:
                raise
            LOGGER.warning(
                "Provider rate limiter unavailable for %s/%s; using local fallback.",
                provider,
                operation,
                exc_info=exc,
            )
            if self.fallback_to_local and interval_seconds > 0:
                self.sleeper(interval_seconds)
            return None
        if reservation.delay_seconds > 0:
            self.sleeper(reservation.delay_seconds)
        return None
