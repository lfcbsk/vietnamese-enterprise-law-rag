from __future__ import annotations

from typing import Any

from src.indexing.tokenize import vietnamese_tokenize
from src.retrieval.schema import RetrievalResult


class BM25Retriever:
    """Retriever từ khóa dùng BM25 index đã được build sẵn."""

    def __init__(self) -> None:
        try:
            from src.indexing.build_bm25_index import load_chunks, load_index
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài rank-bm25. Hãy chạy `python -m pip install -e .` "
                "hoặc `python -m pip install rank-bm25`."
            ) from exc

        self.bm25, self.doc_ids = load_index()
        self.chunk_by_id: dict[str, dict[str, Any]] = {
            chunk["id"]: chunk
            for chunk in load_chunks()
        }

        missing_ids = [
            chunk_id
            for chunk_id in self.doc_ids
            if chunk_id not in self.chunk_by_id
        ]
        if missing_ids:
            raise ValueError(
                "BM25 index không khớp law_chunks.json; "
                "hãy build lại index. Thiếu chunk: "
                f"{missing_ids[:5]}"
            )

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query = query.strip()
        if not query or top_k <= 0:
            return []

        tokens = vietnamese_tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        # Score 0 nghĩa là không có token nào match. Không đưa các tài liệu
        # ngẫu nhiên này vào RRF vì chúng sẽ được cộng điểm như một hit thật.
        ranked = [item for item in ranked if float(item[1]) > 0][:top_k]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                content=self.chunk_by_id[chunk_id]["content"],
                score=float(score),
                rank=rank,
                source="bm25",
                metadata=self.chunk_by_id[chunk_id]["metadata"],
            )
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Alias thống nhất với service/RAG layer."""
        return self.search(query, top_k=top_k)
