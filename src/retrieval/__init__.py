from __future__ import annotations

from typing import Any

from src.retrieval.schema import RetrievalResult

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
]


def __getattr__(name: str) -> Any:
    """Lazy import để schema/fusion không bắt buộc load model và vector DB."""
    if name == "BM25Retriever":
        from src.retrieval.bm25_retriever import BM25Retriever

        return BM25Retriever
    if name == "DenseRetriever":
        from src.retrieval.dense_retriever import DenseRetriever

        return DenseRetriever
    if name == "HybridRetriever":
        from src.retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
