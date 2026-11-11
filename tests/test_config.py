from production_rag.core.config import Settings


def test_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("RERANK_POOL_MAX", "50")

    settings = Settings.from_env()

    assert settings.rerank_pool_max == 50
    assert settings.dense_dim == 768
    assert not hasattr(settings, "service_token")
