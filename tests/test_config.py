import pytest
from pydantic import ValidationError

from grounded_rag.config import Settings


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_SEARCH_MODE", "dense")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")

    settings = _settings()

    assert settings.top_k == 9
    assert settings.search_mode == "dense"
    assert settings.gemini_api_key.get_secret_value() == "sk-test"


def test_blank_values_are_treated_as_unset(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "")
    monkeypatch.setenv("RAG_QDRANT_URL", "  ")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    settings = _settings()

    assert settings.api_key is None
    assert settings.qdrant_url is None
    assert settings.gemini_api_key is None


def test_secrets_are_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "s3cret")

    assert "s3cret" not in repr(_settings())


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        _settings(chunk_size=100, chunk_overlap=100)


def test_unknown_search_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_MODE", "keyword")

    with pytest.raises(ValidationError):
        _settings()
