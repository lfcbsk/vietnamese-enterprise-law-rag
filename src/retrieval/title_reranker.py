"""
Rerank kết quả HybridRetriever theo độ khớp article_title / số Điều.

Đặt file này ở src/retrieval/title_reranker.py.

Cách dùng trong rag_engine.py (_retrieve):
    from src.retrieval.title_reranker import rerank_by_title

    candidates = self.retriever.search(
        query,
        top_k=self.settings.rag_candidate_k,   # lấy nhiều ứng viên hơn, vd 30-40
        candidate_k=self.settings.rag_candidate_k,
    )
    results = rerank_by_title(query, candidates, top_k=self.settings.rag_top_k)
"""
from __future__ import annotations

import re

from src.indexing.tokenize import vietnamese_tokenize
from src.retrieval.schema import RetrievalResult

_EXPLICIT_ARTICLE = re.compile(r"đi[eề]u\s+(\d+[a-zđ]?)", re.IGNORECASE)


def _token_set(text: str) -> set[str]:
    return set(
        vietnamese_tokenize(
            text,
            remove_stopwords=True,
            fold_accents=True,
        )
    )


def rerank_by_title(
    query: str,
    candidates: list[RetrievalResult],
    *,
    top_k: int = 5,
    title_weight: float = 0.15,
    explicit_article_boost: float = 1.0,
) -> list[RetrievalResult]:
    """Re-score các ứng viên hybrid bằng tín hiệu article_title.

    - Cộng thêm điểm tỷ lệ token trùng giữa câu hỏi và article_title
      (Jaccard-lite trên token đã fold dấu + bỏ stopword).
    - Nếu câu hỏi có "Điều N" tường minh và candidate đúng là Điều N,
      cộng điểm rất lớn để luôn đứng đầu.
    - Giữ nguyên fusion score gốc làm nền, không thay thế hoàn toàn -
      tránh việc title match giả (trùng vài từ chung chung) đè lên
      candidate có nội dung khớp tốt nhưng title không trùng.
    """
    if not candidates:
        return []

    query_tokens = _token_set(query)
    explicit_match = _EXPLICIT_ARTICLE.search(query)
    explicit_article_num = (
        explicit_match.group(1).lower() if explicit_match else None
    )

    # Chuẩn hoá fusion score gốc về [0, 1] để cộng dồn với title score
    # có cùng thang đo, tránh việc rrf score (rất nhỏ, ~0.01-0.05) bị
    # title score (0-1) lấn át hoàn toàn.
    max_base = max((c.score for c in candidates), default=1.0) or 1.0

    rescored: list[tuple[float, RetrievalResult]] = []

    for candidate in candidates:
        base = candidate.score / max_base

        title = candidate.metadata.get("article_title", "") or ""
        title_tokens = _token_set(title)

        if query_tokens and title_tokens:
            overlap = len(query_tokens & title_tokens) / len(query_tokens)
        else:
            overlap = 0.0

        boost = 0.0
        if explicit_article_num is not None:
            article_field = str(candidate.metadata.get("article", ""))
            article_num_match = re.search(
                r"(\d+[a-zđ]?)", article_field, re.IGNORECASE
            )
            if (
                article_num_match
                and article_num_match.group(1).lower() == explicit_article_num
            ):
                boost = explicit_article_boost

        final_score = base + title_weight * overlap + boost
        rescored.append((final_score, candidate))

    rescored.sort(key=lambda pair: pair[0], reverse=True)

    reranked: list[RetrievalResult] = []
    for rank, (score, candidate) in enumerate(rescored[:top_k], start=1):
        metadata = dict(candidate.metadata)
        metadata["rerank_score"] = score
        metadata["fusion_score"] = candidate.score
        reranked.append(
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                score=score,
                rank=rank,
                source="hybrid+title_rerank",
                metadata=metadata,
            )
        )

    return reranked