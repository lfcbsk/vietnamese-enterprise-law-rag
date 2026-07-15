from src.retrieval.schema import RetrievalResult
from src.retrieval.title_reranker import rerank_by_title


def _result(article: str, title: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=article,
        content=title,
        score=score,
        rank=1,
        source="hybrid",
        metadata={
            "article": article,
            "article_title": title,
        },
    )


def test_explicit_article_is_boosted_to_first_place() -> None:
    candidates = [
        _result("Điều 120", "Cổ phần phổ thông", 0.02),
        _result("Điều 111", "Công ty cổ phần", 0.01),
    ]

    results = rerank_by_title(
        "Điều 111 quy định gì về công ty cổ phần?",
        candidates,
    )

    assert results[0].metadata["article"] == "Điều 111"
    assert results[0].rank == 1
    assert results[0].metadata["fusion_score"] == 0.01


def test_empty_candidates_return_empty_results() -> None:
    assert rerank_by_title("Điều 1", []) == []
