from typing import Optional

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    llm_provider: str = "anthropic"
    langchain_api_key: str
    github_token: Optional[str] = None
    langchain_tracing_v2: str = "true"
    langchain_project: str = "paperintel"
    postgres_url: str = "postgresql://paperintel:dev_password@localhost:5432/paperintel"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "paper_chunks"
    qdrant_timeout: float = 10.0
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"


settings = Settings()

os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
