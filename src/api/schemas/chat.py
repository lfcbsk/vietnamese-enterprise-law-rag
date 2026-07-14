from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )
    conversation_id: str | None = None


class SourceResponse(BaseModel):
    chunk_id: str
    article: str
    article_title: str
    source: str
    page_start: int
    page_end: int
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    standalone_query: str
    sources: list[SourceResponse]