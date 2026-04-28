import logging
from typing import Optional

import anthropic
import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_anthropic_client: Optional[anthropic.Anthropic] = None


def _provider() -> str:
    return (getattr(settings, "llm_provider", "anthropic") or "anthropic").strip().lower()


def _anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _extract_anthropic_text(response: object) -> tuple[Optional[str], Optional[str]]:
    content = getattr(response, "content", None)
    if not content:
        return None, "Anthropic response returned empty content"
    block = content[0]
    raw = getattr(block, "text", None)
    if not isinstance(raw, str) or not raw.strip():
        return None, "Anthropic response returned non-text or empty content"
    return raw.strip(), None


def _extract_openai_text(payload: dict) -> tuple[Optional[str], Optional[str]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "OpenAI response returned empty choices"
    message = choices[0].get("message", {})
    raw = message.get("content")
    if not isinstance(raw, str) or not raw.strip():
        return None, "OpenAI response returned non-text or empty content"
    return raw.strip(), None


def call_text_llm(
    *,
    requested_model: Optional[str],
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    context_label: str,
) -> tuple[Optional[str], Optional[str]]:
    provider = _provider()

    if provider == "openai":
        model = getattr(settings, "openai_model", "gpt-4o-mini")
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "max_completion_tokens": max_tokens,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            raw, error = _extract_openai_text(response.json())
            if error:
                return None, f"{context_label} failed: {error}"
            logger.info(
                "%s response: %d chars [provider=openai model=%s]",
                context_label,
                len(raw or ""),
                model,
            )
            return raw, None
        except httpx.HTTPStatusError as exc:
            response = exc.response
            body = ""
            headers = {}
            if response is not None:
                try:
                    body = response.text
                except Exception:
                    body = "<unavailable>"
                headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower().startswith("x-ratelimit")
                    or key.lower() in {"retry-after", "x-request-id"}
                }
            logger.exception(
                "%s call failed via OpenAI: status=%s headers=%s body=%s",
                context_label,
                response.status_code if response is not None else "unknown",
                headers,
                body,
            )
            return None, (
                f"{context_label} call failed: status="
                f"{response.status_code if response is not None else 'unknown'} "
                f"body={body}"
            )
        except Exception as exc:
            logger.exception("%s call failed via OpenAI", context_label)
            return None, f"{context_label} call failed: {exc}"

    model = requested_model or settings.haiku_model
    try:
        response = _anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw, error = _extract_anthropic_text(response)
        if error:
            return None, f"{context_label} failed: {error}"
        logger.info(
            "%s response: %d chars [provider=anthropic model=%s]",
            context_label,
            len(raw or ""),
            model,
        )
        return raw, None
    except Exception as exc:
        logger.exception("%s call failed via Anthropic", context_label)
        return None, f"{context_label} call failed: {exc}"
