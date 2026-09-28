"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config for API keys and service settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq LLM (Step 4+)
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # Tavily web search (Step 6+)
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")

    # LangSmith tracing (Step 10+)
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="us-stock-research",
        validation_alias="LANGCHAIN_PROJECT",
    )

    # SEC EDGAR fair access (Step 1+)
    sec_edgar_user_agent: str = Field(default="", validation_alias="SEC_EDGAR_USER_AGENT")

    # Paths
    data_dir: str = Field(default="data")
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    chroma_dir: str = Field(default="data/chroma")


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
