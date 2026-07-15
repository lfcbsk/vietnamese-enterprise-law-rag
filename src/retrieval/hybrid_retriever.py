from __future__ import annotations

from typing import Any, Protocol

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.schema import RetrievalResult


DEFAULT_SOURCE_WEIGHTS = {
    "dense": 0.9,
    "bm25": 0.1,
}


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 20,
        **kwargs: Any,
    ) -> list[RetrievalResult]: ...


class HybridRetriever:
    """Kết hợp lexical BM25 và semantic dense retrieval bằng weighted RRF."""

    def __init__(
        self,
        *,
        bm25_retriever: Retriever | None = None,
        dense_retriever: Retriever | None = None,
        rrf_k: int = 60,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.bm25 = bm25_retriever or BM25Retriever()
        self.dense = dense_retriever or DenseRetriever()
        self.rrf_k = rrf_k
        self.source_weights = (
            dict(source_weights)
            if source_weights is not None
            else dict(DEFAULT_SOURCE_WEIGHTS)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query = query.strip()
        if not query or top_k <= 0:
            return []

        candidate_k = max(candidate_k, top_k)
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(
            query,
            top_k=candidate_k,
            where=where,
        )

        return reciprocal_rank_fusion(
            [bm25_results, dense_results],
            top_k=top_k,
            rrf_k=self.rrf_k,
            source_weights=self.source_weights,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return self.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            where=where,
        )
