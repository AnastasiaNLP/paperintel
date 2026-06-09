import os

from config.settings_model import Settings


settings = Settings()

os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
