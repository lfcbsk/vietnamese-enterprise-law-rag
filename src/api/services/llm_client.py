from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.api.config import Settings


def create_llm(
    settings: Settings,
) -> ChatOpenAI:
    kwargs = {
        "model": settings.llm_model,
        "api_key": (
            settings.llm_api_key.get_secret_value()
        ),
        "temperature": settings.llm_temperature,
        "timeout": 60,
        "max_retries": 2,
    }

    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url

    return ChatOpenAI(**kwargs)