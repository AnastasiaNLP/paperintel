from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    langchain_api_key: str
    langchain_tracing_v2: str = "true"
    langchain_project: str = "paperintel"
    postgres_url: str = "postgresql://paperintel:dev_password@localhost:5432/paperintel"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()