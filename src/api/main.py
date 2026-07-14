from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.config import get_settings
from src.api.routers.chat import (
    router as chat_router,
)
from src.api.routers.health import (
    router as health_router,
)
from src.api.services.llm_client import (
    create_llm,
)
from src.api.services.rag_engine import (
    RAGEngine,
)
from src.retrieval import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    llm = create_llm(settings)
    retriever = HybridRetriever()

    engine = RAGEngine(
        llm=llm,
        retriever=retriever,
        settings=settings,
    )

    app.state.rag_engine = engine

    yield

    engine.close()


app = FastAPI(
    title="Vietnamese Enterprise Law RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)