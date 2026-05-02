from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field


ErrorSeverity: TypeAlias = Literal["info", "warning", "error", "fatal"]


class ErrorCodes:
    PAPER_ERROR = "PAPER_ERROR"
    FATAL_ERROR = "FATAL_ERROR"
    WARNING = "WARNING"


class StructuredError(BaseModel):
    code: str
    message: str
    node: str | None = None
    agent: str | None = None
    severity: ErrorSeverity = "error"
    recoverable: bool = True
    paper_id: str | None = None
    session_id: str | None = None
    agent_run_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


ErrorLike: TypeAlias = StructuredError | str


def make_error(
    code: str,
    message: str,
    *,
    node: str | None = None,
    agent: str | None = None,
    severity: ErrorSeverity = "error",
    recoverable: bool = True,
    paper_id: str | None = None,
    session_id: str | None = None,
    agent_run_id: str | None = None,
    **details: Any,
) -> StructuredError:
    return StructuredError(
        code=code,
        message=message,
        node=node,
        agent=agent,
        severity=severity,
        recoverable=recoverable,
        paper_id=paper_id,
        session_id=session_id,
        agent_run_id=agent_run_id,
        details=details,
    )


def normalize_error(
    error: ErrorLike,
    *,
    default_code: str = ErrorCodes.WARNING,
    default_node: str | None = None,
) -> StructuredError:
    if isinstance(error, StructuredError):
        return error

    return StructuredError(
        code=default_code,
        message=str(error),
        node=default_node,
        severity="warning",
        recoverable=True,
    )


def normalize_errors(errors: list[ErrorLike]) -> list[StructuredError]:
    return [normalize_error(error) for error in errors]


def error_message(error: ErrorLike) -> str:
    if isinstance(error, StructuredError):
        return error.message
    return str(error)


def error_messages(errors: list[ErrorLike]) -> list[str]:
    return [error_message(error) for error in errors]
