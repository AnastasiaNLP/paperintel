import sys
from pathlib import Path

import os


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")


def assert_provider_failure_logged(
    message: str,
    *,
    provider: str,
    operation: str,
    failure_class: str,
    retryable: bool | None = None,
    forbidden: str | None = None,
) -> None:
    assert "event=provider.failure" in message
    assert f'provider="{provider}"' in message
    assert f'operation="{operation}"' in message
    assert f'failure_class="{failure_class}"' in message
    if retryable is not None:
        assert f"retryable={'true' if retryable else 'false'}" in message
    if forbidden is not None:
        assert forbidden not in message
