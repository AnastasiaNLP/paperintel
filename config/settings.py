from typing import Optional

from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    langchain_api_key: str
    github_token: Optional[str] = None
    langchain_tracing_v2: str = "true"
    langchain_project: str = "paperintel"
    postgres_url: str = "postgresql://paperintel:dev_password@localhost:5432/paperintel"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)