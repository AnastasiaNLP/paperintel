from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import httpx

from services.observability import emit_event

logger = logging.getLogger(__name__)


class FailureClass(StrEnum):
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_NOT_FOUND = "provider_not_found"
    INVALID_INPUT = "invalid_input"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CANCELED = "canceled"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ClassifiedFailure:
    provider: str
    operation: str
    failure_class: FailureClass
    retryable: bool
    breaker_failure: bool
    degradation_allowed: bool = False
    http_status: int | None = None


def classify_provider_exception(
    provider: str,
    operation: str,
    exc: BaseException,
    *,
    not_found_exception_types: Iterable[type[BaseException]] = (),
    invalid_input_exception_types: Iterable[type[BaseException]] = (),
    dependency_unavailable_exception_types: Iterable[type[BaseException]] = (),
    dependency_not_found_exception_types: Iterable[type[BaseException]] = (),
    circuit_open_exception_types: Iterable[type[BaseException]] = (),
    canceled_exception_types: Iterable[type[BaseException]] = (),
    degradation_allowed: bool = False,
    default_retryable: bool = False,
    default_breaker_failure: bool = False,
) -> ClassifiedFailure:
    not_found_types = tuple(not_found_exception_types)
    invalid_input_types = tuple(invalid_input_exception_types)
    dependency_unavailable_types = tuple(dependency_unavailable_exception_types)
    dependency_not_found_types = tuple(dependency_not_found_exception_types)
    circuit_open_types = tuple(circuit_open_exception_types)
    canceled_types = tuple(canceled_exception_types)

    if canceled_types and isinstance(exc, canceled_types):
        return _classified(
            provider,
            operation,
            FailureClass.CANCELED,
            retryable=False,
            breaker_failure=False,
        )
    if not_found_types and isinstance(exc, not_found_types):
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_NOT_FOUND,
            retryable=False,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
        )
    if dependency_not_found_types and isinstance(exc, dependency_not_found_types):
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_NOT_FOUND,
            retryable=False,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
        )
    if dependency_unavailable_types and isinstance(exc, dependency_unavailable_types):
        return _classified(
            provider,
            operation,
            FailureClass.DEPENDENCY_UNAVAILABLE,
            retryable=True,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
        )
    if circuit_open_types and isinstance(exc, circuit_open_types):
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_UNAVAILABLE,
            retryable=False,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
        )
    if invalid_input_types and isinstance(exc, invalid_input_types):
        return _classified(
            provider,
            operation,
            FailureClass.INVALID_INPUT,
            retryable=False,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_http_status(
            provider,
            operation,
            exc.response.status_code,
            degradation_allowed=degradation_allowed,
        )
    if isinstance(exc, httpx.TimeoutException):
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_TIMEOUT,
            retryable=True,
            breaker_failure=True,
            degradation_allowed=degradation_allowed,
        )
    if isinstance(exc, httpx.TransportError):
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_UNAVAILABLE,
            retryable=True,
            breaker_failure=True,
            degradation_allowed=degradation_allowed,
        )
    return _classified(
        provider,
        operation,
        FailureClass.INTERNAL_ERROR,
        retryable=default_retryable,
        breaker_failure=default_breaker_failure,
        degradation_allowed=degradation_allowed,
    )


def classify_http_status(
    provider: str,
    operation: str,
    status_code: int,
    *,
    degradation_allowed: bool = False,
) -> ClassifiedFailure:
    if status_code == 429:
        return _classified(
            provider,
            operation,
            FailureClass.RATE_LIMITED,
            retryable=True,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
            http_status=status_code,
        )
    if status_code == 404:
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_NOT_FOUND,
            retryable=False,
            breaker_failure=False,
            degradation_allowed=degradation_allowed,
            http_status=status_code,
        )
    if status_code == 408:
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_TIMEOUT,
            retryable=True,
            breaker_failure=True,
            degradation_allowed=degradation_allowed,
            http_status=status_code,
        )
    if status_code >= 500:
        return _classified(
            provider,
            operation,
            FailureClass.PROVIDER_UNAVAILABLE,
            retryable=True,
            breaker_failure=True,
            degradation_allowed=degradation_allowed,
            http_status=status_code,
        )
    return _classified(
        provider,
        operation,
        FailureClass.INVALID_INPUT,
        retryable=False,
        breaker_failure=False,
        degradation_allowed=degradation_allowed,
        http_status=status_code,
    )


def _classified(
    provider: str,
    operation: str,
    failure_class: FailureClass,
    *,
    retryable: bool,
    breaker_failure: bool,
    degradation_allowed: bool = False,
    http_status: int | None = None,
) -> ClassifiedFailure:
    return ClassifiedFailure(
        provider=provider,
        operation=operation,
        failure_class=failure_class,
        retryable=retryable,
        breaker_failure=breaker_failure,
        degradation_allowed=degradation_allowed,
        http_status=http_status,
    )


def emit_provider_failure(
    failure: ClassifiedFailure,
    *,
    retry_after_seconds: float | None = None,
) -> None:
    emit_event(
        logger,
        "provider.failure",
        provider=failure.provider,
        operation=failure.operation,
        failure_class=failure.failure_class.value,
        retryable=failure.retryable,
        retry_after_seconds=retry_after_seconds,
    )
