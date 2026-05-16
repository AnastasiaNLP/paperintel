from typing import Any

from sqlalchemy import text

from models.api import HealthStatus


class HealthChecker:
    def __init__(
        self,
        *,
        session_factory: Any | None = None,
        qdrant_store: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.qdrant_store = qdrant_store
        self.settings = settings

    def check(self) -> HealthStatus:
        checks = {
            "postgres": self._check_postgres(),
            "qdrant": self._check_qdrant(),
            "llm_provider": self._check_llm_provider(),
            "openai_embeddings": self._check_openai_embeddings(),
        }
        healthy = (
            checks["postgres"] == "ok"
            and checks["qdrant"] == "ok"
            and checks["llm_provider"] == "configured"
            and checks["openai_embeddings"] == "configured"
        )
        return HealthStatus(healthy=healthy, checks=checks)

    def _check_postgres(self) -> str:
        if self.session_factory is None:
            return "not_configured"
        try:
            with self.session_factory() as db:
                db.execute(text("SELECT 1"))
            return "ok"
        except Exception as exc:
            return f"error:{type(exc).__name__}"

    def _check_qdrant(self) -> str:
        if self.qdrant_store is None:
            return "not_configured"
        try:
            self.qdrant_store.check_connection()
            return "ok"
        except Exception as exc:
            return f"error:{type(exc).__name__}"

    def _check_llm_provider(self) -> str:
        if self.settings is None:
            return "not_configured"
        anthropic_key = getattr(self.settings, "anthropic_api_key", "") or ""
        openai_key = getattr(self.settings, "openai_api_key", "") or ""
        if anthropic_key.strip() or openai_key.strip():
            return "configured"
        return "missing_api_key"

    def _check_openai_embeddings(self) -> str:
        if self.settings is None:
            return "not_configured"
        openai_key = getattr(self.settings, "openai_api_key", "") or ""
        return "configured" if openai_key.strip() else "missing_api_key"
