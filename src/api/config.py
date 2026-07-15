from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: SecretStr
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: str | None = None
    llm_temperature: float = 0.0

    rag_top_k: int = 5
    rag_candidate_k: int = 40

    chat_db_path: Path = Path(
        "data/chat/checkpoints.sqlite"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
