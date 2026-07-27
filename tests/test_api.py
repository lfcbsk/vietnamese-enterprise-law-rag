from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import main as api_main
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
    app.state.initialization_status = "ready"
    app.include_router(health_router)
    app.include_router(chat_router)
    return TestClient(app), engine


def test_health_endpoint_reports_ready() -> None:
    client, _ = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_main_app_mounts_gradio_ui() -> None:
    route_paths = {
        route.path
        for route in api_main.app.routes
        if hasattr(route, "path")
    }

    assert "/app" in route_paths


def test_readiness_endpoint_reports_initializing() -> None:
    client, _ = _client()
    client.app.state.initialization_status = "initializing"

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "initializing"}


def test_readiness_endpoint_reports_initialization_failure() -> None:
    client, _ = _client()
    client.app.state.initialization_status = "failed"
    client.app.state.initialization_error = "RuntimeError"

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "failed",
        "error": "RuntimeError",
    }


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


def test_chat_endpoint_rejects_requests_while_initializing() -> None:
    client, engine = _client()
    client.app.state.initialization_status = "initializing"

    response = client.post(
        "/chat",
        json={"question": "Điều 17 quy định gì?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Mô hình đang khởi tạo. Vui lòng thử lại sau."
    }
    assert engine.calls == []


def test_main_lifespan_does_not_wait_for_retriever(
    monkeypatch,
) -> None:
    release_retriever = threading.Event()
    engine_closed = threading.Event()

    def blocking_retriever() -> object:
        release_retriever.wait(timeout=5)
        return object()

    class EngineStub:
        def close(self) -> None:
            engine_closed.set()

    monkeypatch.setattr(api_main, "get_settings", lambda: object())
    monkeypatch.setattr(api_main, "create_llm", lambda settings: object())
    monkeypatch.setattr(api_main, "HybridRetriever", blocking_retriever)
    monkeypatch.setattr(api_main, "RAGEngine", lambda **kwargs: EngineStub())

    with TestClient(api_main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health/ready").status_code == 503

        release_retriever.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            response = client.get("/health/ready")
            if response.status_code == 200:
                break
            time.sleep(0.01)

        assert response.json() == {"status": "ready"}

    assert engine_closed.is_set()
