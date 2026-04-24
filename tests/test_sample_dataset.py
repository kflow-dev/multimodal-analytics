from src.sample_dataset import provider_status


def test_provider_status_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_ENGINE_ID", raising=False)

    status = provider_status()

    assert status["openai_web_search"] is False
    assert status["google_custom_search"] is False
