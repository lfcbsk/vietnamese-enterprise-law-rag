from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.api.services.rag_engine import RAGEngine
from src.retrieval.article_lookup import (
    ArticleLookup,
    extract_article_numbers,
)
from src.retrieval.schema import RetrievalResult


def _lookup() -> ArticleLookup:
    chunks = [
        {
            "id": "law_dieu_17",
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp 2020",
            "chapter": "Chương II",
            "article": "Điều 17",
            "article_title": "Quyền thành lập doanh nghiệp",
            "content": "Điều 17. Quyền thành lập doanh nghiệp...",
            "metadata": {
                "article": "Điều 17",
                "article_title": "Quyền thành lập doanh nghiệp",
            },
        },
        {
            "id": "law_dieu_111",
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp 2020",
            "chapter": "Chương V",
            "article": "Điều 111",
            "article_title": "Công ty cổ phần",
            "content": "Điều 111. Công ty cổ phần...",
            "metadata": {
                "article": "Điều 111",
                "article_title": "Công ty cổ phần",
            },
        },
    ]
    lookup = object.__new__(ArticleLookup)
    lookup._articles = {
        "17": [chunks[0]],
        "111": [chunks[1]],
    }
    return lookup


def test_extracts_accented_unaccented_and_unique_article_numbers() -> None:
    assert extract_article_numbers(
        "Khoản 2 Điều 17, dieu 111 và Điều 17"
    ) == ["17", "111"]


def test_direct_lookup_returns_requested_articles_in_query_order() -> None:
    lookup = _lookup()

    results = lookup.search("So sánh Điều 111 với Điều 17", top_k=5)

    assert [result.metadata["article"] for result in results] == [
        "Điều 111",
        "Điều 17",
    ]
    assert all(result.source == "article_lookup" for result in results)
    assert [result.rank for result in results] == [1, 2]


def test_lookup_returns_empty_for_semantic_or_unknown_article() -> None:
    lookup = _lookup()

    assert lookup.search("Quyền của cổ đông là gì?") == []
    assert lookup.search("Điều 999 quy định gì?") == []


def test_article_lookup_success_routes_directly_to_generate() -> None:
    direct_result = RetrievalResult(
        chunk_id="law_dieu_111",
        content="Điều 111. Công ty cổ phần...",
        score=1.0,
        rank=1,
        source="article_lookup",
        metadata={
            "article": "Điều 111",
            "article_title": "Công ty cổ phần",
        },
    )

    class DirectLookupStub:
        def search(self, query: str, top_k: int) -> list[RetrievalResult]:
            return [direct_result]

    engine = object.__new__(RAGEngine)
    engine.article_lookup = DirectLookupStub()
    engine.settings = SimpleNamespace(rag_top_k=5, rag_candidate_k=40)

    response = engine._article_lookup(
        {"standalone_query": "Điều 111 quy định gì?"}
    )

    assert response["sources"][0]["article"] == "Điều 111"
    assert "Điều 111. Công ty cổ phần" in response["context"]
    assert response["lookup_succeeded"] is True
    assert engine._route_retrieval(response) == "direct"


def test_article_lookup_miss_routes_to_hybrid_retrieval() -> None:
    class EmptyLookupStub:
        def search(self, query: str, top_k: int) -> list[RetrievalResult]:
            return []

    engine = object.__new__(RAGEngine)
    engine.article_lookup = EmptyLookupStub()
    engine.settings = SimpleNamespace(rag_top_k=5)

    response = engine._article_lookup(
        {"standalone_query": "Quyền của cổ đông là gì?"}
    )

    assert response == {
        "context": "",
        "sources": [],
        "lookup_succeeded": False,
    }
    assert engine._route_retrieval(response) == "hybrid"


def test_structural_follow_up_reuses_previous_article_without_llm() -> None:
    engine = object.__new__(RAGEngine)

    state = {
        "messages": [
            HumanMessage(content="Điều 111 quy định gì?"),
            AIMessage(content="Điều 111 quy định về công ty cổ phần."),
            HumanMessage(content="Khoản 2 nói gì?"),
        ],
        "standalone_query": "Điều 111 quy định gì?",
        "sources": [{"article": "Điều 111"}],
    }

    assert engine._route_query_rewrite(state) == "structural"
    response = engine._rewrite_structural_query(
        state
    )

    assert response == {
        "standalone_query": "Khoản 2 nói gì? (thuộc Điều 111)"
    }


def test_semantic_follow_up_is_rewritten_from_short_memory() -> None:
    class RewriteLLMStub:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def invoke(self, messages: list[object]) -> AIMessage:
            self.messages = messages
            return AIMessage(
                content="Quyền của cổ đông trong công ty cổ phần là gì?"
            )

    engine = object.__new__(RAGEngine)
    engine.llm = RewriteLLMStub()

    state = {
        "messages": [
            HumanMessage(content="Điều 111 quy định gì?"),
            AIMessage(content="Điều 111 quy định về công ty cổ phần."),
            HumanMessage(content="Vậy quyền cổ đông thì sao?"),
        ],
        "standalone_query": "Điều 111 quy định gì?",
        "sources": [{"article": "Điều 111"}],
    }

    assert engine._route_query_rewrite(state) == "llm"
    response = engine._rewrite_query_with_llm(state)

    assert response == {
        "standalone_query": "Quyền của cổ đông trong công ty cổ phần là gì?"
    }
    assert len(engine.llm.messages) == 4
    assert isinstance(engine.llm.messages[0], SystemMessage)
    assert engine.llm.messages[-1].content == "Vậy quyền cổ đông thì sao?"
