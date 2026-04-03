from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0       # deterministic for agent reasoning

    # External API
    rest_countries_base_url: str = "https://restcountries.com/v3.1"
    http_timeout_seconds: float = 10.0
    http_max_retries: int = 3

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_question_length: int = 500


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — reads .env once at startup."""
    return Settings()