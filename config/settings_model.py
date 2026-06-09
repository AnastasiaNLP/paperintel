from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.retrieval import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL


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
    blob_storage_enabled: bool = True
    blob_s3_endpoint_url: str = "http://localhost:9000"
    blob_s3_region: str = "us-east-1"
    blob_s3_bucket: str = "paperintel"
    blob_s3_access_key_id: str = "paperintel"
    blob_s3_secret_access_key: str = "paperintel_dev_password"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    openai_embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    openai_embedding_timeout: float = 30.0

    @field_validator("openai_embedding_dimensions")
    @classmethod
    def openai_embedding_dimensions_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("openai_embedding_dimensions must be positive")
        return value

    @field_validator("openai_embedding_timeout")
    @classmethod
    def openai_embedding_timeout_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("openai_embedding_timeout must be positive")
        return value
