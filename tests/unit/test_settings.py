import pytest
from pydantic import ValidationError

from config.settings_model import Settings
from models.retrieval import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL


def _settings(**overrides):
    values = {
        "anthropic_api_key": "anthropic-key",
        "openai_api_key": "openai-key",
        "langchain_api_key": "langchain-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_settings_embedding_defaults_match_retrieval_contract():
    settings = _settings()

    assert settings.openai_embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.openai_embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.openai_embedding_timeout == 30.0


def test_settings_rejects_invalid_embedding_runtime_values():
    with pytest.raises(ValidationError, match="openai_embedding_dimensions"):
        _settings(openai_embedding_dimensions=0)

    with pytest.raises(ValidationError, match="openai_embedding_timeout"):
        _settings(openai_embedding_timeout=0)
