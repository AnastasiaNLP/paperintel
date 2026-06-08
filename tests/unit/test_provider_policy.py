import httpx

from services.provider_policy import FailureClass, classify_provider_exception
from tools.circuit_breaker import CircuitBreakerOpenError


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://provider.example/test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("provider error", request=request, response=response)


def test_classifies_rate_limit_as_retryable_not_breaker_failure():
    classified = classify_provider_exception("arxiv", "search", _http_error(429))

    assert classified.failure_class == FailureClass.RATE_LIMITED
    assert classified.retryable is True
    assert classified.breaker_failure is False
    assert classified.http_status == 429


def test_classifies_5xx_as_retryable_breaker_failure():
    classified = classify_provider_exception("arxiv", "search", _http_error(503))

    assert classified.failure_class == FailureClass.PROVIDER_UNAVAILABLE
    assert classified.retryable is True
    assert classified.breaker_failure is True


def test_classifies_timeout_as_retryable_breaker_failure():
    classified = classify_provider_exception(
        "semantic_scholar",
        "paper_enrichment",
        httpx.TimeoutException("timed out"),
    )

    assert classified.failure_class == FailureClass.PROVIDER_TIMEOUT
    assert classified.retryable is True
    assert classified.breaker_failure is True


def test_classifies_not_found_and_invalid_input_as_non_retryable():
    not_found = classify_provider_exception("arxiv", "metadata", _http_error(404))
    invalid = classify_provider_exception("arxiv", "metadata", ValueError("bad input"))

    assert not_found.failure_class == FailureClass.PROVIDER_NOT_FOUND
    assert not_found.retryable is False
    assert invalid.failure_class == FailureClass.INTERNAL_ERROR
    assert invalid.retryable is False


def test_local_exception_types_override_default_policy():
    class PaperNotFound(ValueError):
        pass

    classified = classify_provider_exception(
        "arxiv",
        "metadata",
        PaperNotFound("missing"),
        not_found_exception_types=(PaperNotFound,),
        invalid_input_exception_types=(ValueError,),
        default_retryable=True,
        default_breaker_failure=True,
    )

    assert classified.failure_class == FailureClass.PROVIDER_NOT_FOUND
    assert classified.retryable is False
    assert classified.breaker_failure is False


def test_circuit_open_is_not_retryable_or_breaker_failure():
    classified = classify_provider_exception(
        "arxiv",
        "metadata",
        CircuitBreakerOpenError("arxiv", 30.0),
        circuit_open_exception_types=(CircuitBreakerOpenError,),
        default_retryable=True,
        default_breaker_failure=True,
    )

    assert classified.failure_class == FailureClass.PROVIDER_UNAVAILABLE
    assert classified.retryable is False
    assert classified.breaker_failure is False
