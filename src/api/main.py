from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from contextlib import suppress
from typing import Any
from uuid import uuid4

import gradio as gr
from fastapi import FastAPI

from src.api.config import get_settings
from src.api.routers.chat import router as chat_router
from src.api.routers.health import router as health_router
from src.api.services.llm_client import create_llm
from src.api.services.rag_engine import RAGEngine
from src.app.app import CSS
from src.app.app import create_demo
from src.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


async def _load_engine(app: FastAPI) -> None:
    """Build the expensive retrieval engine without delaying socket startup."""
    logger.info("RAG engine initialization started")
    try:
        settings = get_settings()
        llm = create_llm(settings)
        retriever = await asyncio.to_thread(HybridRetriever)
        engine = RAGEngine(
            llm=llm,
            retriever=retriever,
            settings=settings,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        app.state.initialization_status = "failed"
        app.state.initialization_error = type(error).__name__
        logger.exception("RAG engine initialization failed")
        return

    app.state.rag_engine = engine
    app.state.initialization_error = None
    app.state.initialization_status = "ready"
    logger.info("RAG engine initialization complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_engine = None
    app.state.initialization_error = None
    app.state.initialization_status = "initializing"
    initialization_task = asyncio.create_task(_load_engine(app))

    yield

    initialization_task.cancel()
    with suppress(asyncio.CancelledError):
        await initialization_task

    engine = app.state.rag_engine
    if engine is not None:
        engine.close()


def create_app() -> FastAPI:
    api = FastAPI(
        title="Vietnamese Enterprise Law RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    @api.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "name": "Vietnamese Enterprise Law RAG",
            "app": "/app",
            "health": "/health",
            "docs": "/docs",
            "chat": "POST /chat",
        }

    api.include_router(health_router)
    api.include_router(chat_router)

    def ui_chat(
        question: str,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        status = getattr(api.state, "initialization_status", "initializing")
        engine = getattr(api.state, "rag_engine", None)
        if status != "ready" or engine is None:
            raise RuntimeError("RAG engine is not ready")

        active_conversation_id = conversation_id or str(uuid4())
        return engine.ask(
            conversation_id=active_conversation_id,
            question=question,
        )

    def ui_readiness() -> tuple[bool, str]:
        status = getattr(api.state, "initialization_status", "initializing")
        if status == "ready":
            return True, "Mô hình đã sẵn sàng."
        if status == "failed":
            return False, "Khởi tạo mô hình thất bại. Vui lòng kiểm tra log."
        return False, "Mô hình đang khởi tạo, vui lòng chờ một chút."

    demo = create_demo(ui_chat, ui_readiness)
    return gr.mount_gradio_app(
        api,
        demo,
        path="/app",
        theme=gr.themes.Soft(),
        css=CSS,
        footer_links=[],
        show_error=False,
    )


app = create_app()
