from stock_research.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings()
    assert settings.langchain_project == "us-stock-research"
    assert settings.data_dir == "data"
    assert settings.chroma_dir == "data/chroma"


def test_get_settings():
    assert isinstance(get_settings(), Settings)
