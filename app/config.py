"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_database_url(url: str) -> str:
    """Force the psycopg (v3) driver for Postgres URLs from Neon/Supabase/Render."""
    url = url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


APP_VERSION = "1.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./learnai.db"
    groq_api_key: str = ""
    groq_model_main: str = "openai/gpt-oss-120b"
    groq_model_fast: str = "openai/gpt-oss-20b"
    groq_reasoning_effort: str = "low"
    allowed_origins: str = "http://localhost:5173,http://localhost:8080"
    enable_docs: bool = True
    verify_questions: bool = True
    rate_limit_llm_per_min: int = 20  # tutor_chat, generate_questions, assessment; 0 disables
    rate_limit_other_per_min: int = 120  # every other action; 0 disables
    db_schema: str = ""  # Postgres only: run in this schema (used by the test suite for isolation)
    tutor_support_text: str = (
        "If you're feeling overwhelmed, please reach out to someone you trust or call Tele-MANAS at 14416 (India)."
    )

    @property
    def sqlalchemy_url(self) -> str:
        return normalize_database_url(self.database_url)

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return ["*"] if "*" in origins else origins

    @property
    def db_type(self) -> str:
        return "postgresql" if self.sqlalchemy_url.startswith("postgresql") else "sqlite"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
