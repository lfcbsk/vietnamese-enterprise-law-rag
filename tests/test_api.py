from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.chat import router as chat_router
from src.api.routers.health import router as health_router


class RAGEngineStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def ask(self, conversation_id: str, question: str) -> dict[str, Any]:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "question": question,
            }
        )
        return {
            "conversation_id": conversation_id,
            "answer": "Câu trả lời kiểm thử.",
            "standalone_query": question,
            "sources": [
                {
                    "chunk_id": "law_dieu_17",
                    "article": "Điều 17",
                    "article_title": "Quyền thành lập doanh nghiệp",
                    "source": "article_lookup",
                    "page_start": 12,
                    "page_end": 13,
                    "score": 1.0,
                }
            ],
        }


def _client() -> tuple[TestClient, RAGEngineStub]:
    app = FastAPI()
    engine = RAGEngineStub()
    app.state.rag_engine = engine
    app.include_router(health_router)
    app.include_router(chat_router)
    return TestClient(app), engine


def test_health_endpoint_reports_ready() -> None:
    client, _ = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_response_contract() -> None:
    client, engine = _client()

    response = client.post(
        "/chat",
        json={
            "question": "  Điều 17 quy định gì?  ",
            "conversation_id": "conversation-1",
        },
    )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "conversation_id": "conversation-1",
            "question": "Điều 17 quy định gì?",
        }
    ]
    assert response.json() == {
        "conversation_id": "conversation-1",
        "answer": "Câu trả lời kiểm thử.",
        "standalone_query": "Điều 17 quy định gì?",
        "sources": [
            {
                "chunk_id": "law_dieu_17",
                "article": "Điều 17",
                "article_title": "Quyền thành lập doanh nghiệp",
                "source": "article_lookup",
                "page_start": 12,
                "page_end": 13,
                "score": 1.0,
            }
        ],
    }


def test_chat_endpoint_rejects_an_empty_question() -> None:
    client, engine = _client()

    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422
    assert engine.calls == []
