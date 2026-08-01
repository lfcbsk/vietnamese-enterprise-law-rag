from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)

from src.api.config import Settings
from src.generation.context_builder import (
    build_context,
    serialize_sources,
)
from src.generation.prompts import (
    LEGAL_SYSTEM_PROMPT,
    REWRITE_QUERY_PROMPT,
)
from src.memory.checkpointer import (
    create_sqlite_checkpointer,
)
from src.retrieval import HybridRetriever
from src.retrieval.article_lookup import ArticleLookup, extract_article_numbers
from src.retrieval.title_reranker import rerank_by_title


_STRUCTURAL_FOLLOW_UP = re.compile(
    r"\b(khoản|điểm|điều này)\b",
    re.IGNORECASE,
)

_SHORT_MEMORY_MESSAGE_LIMIT = 8


class RAGState(MessagesState):
    standalone_query: str
    context: str
    sources: list[dict[str, Any]]
    lookup_succeeded: bool


class RAGEngine:
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        retriever: HybridRetriever,
        settings: Settings,
        article_lookup: ArticleLookup | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        self.article_lookup = article_lookup or ArticleLookup()

        (
            self.checkpointer,
            self._database_connection,
        ) = create_sqlite_checkpointer(
            settings.chat_db_path
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node(
            "use_original_query",
            self._use_original_query,
        )
        builder.add_node(
            "rewrite_structural_query",
            self._rewrite_structural_query,
        )
        builder.add_node(
            "rewrite_query_with_llm",
            self._rewrite_query_with_llm,
        )
        builder.add_node(
            "article_lookup",
            self._article_lookup,
        )
        builder.add_node(
            "hybrid_retrieve",
            self._hybrid_retrieve,
        )
        builder.add_node(
            "generate",
            self._generate,
        )

        builder.add_conditional_edges(
            START,
            self._route_query_rewrite,
            {
                "original": "use_original_query",
                "structural": "rewrite_structural_query",
                "llm": "rewrite_query_with_llm",
            },
        )
        builder.add_edge("use_original_query", "article_lookup")
        builder.add_edge("rewrite_structural_query", "article_lookup")
        builder.add_edge("rewrite_query_with_llm", "article_lookup")
        builder.add_conditional_edges(
            "article_lookup",
            self._route_retrieval,
            {
                "direct": "generate",
                "hybrid": "hybrid_retrieve",
            },
        )
        builder.add_edge("hybrid_retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile(
            checkpointer=self.checkpointer
        )

    @staticmethod
    def _human_messages(state: RAGState) -> list[HumanMessage]:
        return [
            message
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]

    @classmethod
    def _current_query(cls, state: RAGState) -> str:
        return str(cls._human_messages(state)[-1].content).strip()

    def _route_query_rewrite(self, state: RAGState) -> str:
        human_messages = self._human_messages(state)
        query = str(human_messages[-1].content).strip()

        if len(human_messages) == 1 or extract_article_numbers(query):
            return "original"

        has_structural_reference = bool(_STRUCTURAL_FOLLOW_UP.search(query))
        has_previous_article = any(
            source.get("article") for source in state.get("sources", [])
        )
        if has_structural_reference and has_previous_article:
            return "structural"

        return "llm"

    def _use_original_query(
        self,
        state: RAGState,
    ) -> dict[str, str]:
        return {"standalone_query": self._current_query(state)}

    def _rewrite_structural_query(
        self,
        state: RAGState,
    ) -> dict[str, str]:
        query = self._current_query(state)
        previous_article = next(
            (
                str(source["article"]).strip()
                for source in state.get("sources", [])
                if source.get("article")
            ),
            "",
        )

        return {"standalone_query": f"{query} (thuộc {previous_article})"}

    def _rewrite_query_with_llm(
        self,
        state: RAGState,
    ) -> dict[str, str]:
        query = self._current_query(state)
        recent_messages = state["messages"][-_SHORT_MEMORY_MESSAGE_LIMIT:]
        response = self.llm.invoke(
            [
                SystemMessage(content=REWRITE_QUERY_PROMPT),
                *recent_messages,
            ]
        )
        rewritten_query = str(response.content).strip()
        if rewritten_query:
            query = rewritten_query

        return {"standalone_query": query}

    def _article_lookup(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        query = state["standalone_query"]
        results = self.article_lookup.search(
            query,
            top_k=self.settings.rag_top_k,
        )

        return {
            "context": build_context(results),
            "sources": serialize_sources(results),
            "lookup_succeeded": bool(results),
        }

    @staticmethod
    def _route_retrieval(state: RAGState) -> str:
        if state.get("lookup_succeeded", False):
            return "direct"
        return "hybrid"

    def _hybrid_retrieve(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        query = state["standalone_query"]
        candidates = self.retriever.search(
            query,
            top_k=self.settings.rag_candidate_k,
            candidate_k=self.settings.rag_candidate_k,
        )
        results = rerank_by_title(
            query,
            candidates,
            top_k=self.settings.rag_top_k,
        )

        return {
            "context": build_context(results),
            "sources": serialize_sources(results),
        }

    def _generate(
        self,
        state: RAGState,
    ) -> dict[str, list[AIMessage]]:
        context = state.get("context", "")

        system_message = SystemMessage(
            content=LEGAL_SYSTEM_PROMPT.format(
                context=context
            )
        )

        # Chỉ đưa một số turn gần nhất vào LLM.
        # Toàn bộ state vẫn được checkpointer lưu.
        recent_messages = state["messages"][-_SHORT_MEMORY_MESSAGE_LIMIT:]

        response = self.llm.invoke(
            [
                system_message,
                *recent_messages,
            ]
        )

        return {
            "messages": [
                AIMessage(
                    content=str(response.content)
                )
            ]
        }

    def ask(
        self,
        *,
        conversation_id: str,
        question: str,
    ) -> dict[str, Any]:
        config = {
            "configurable": {
                "thread_id": conversation_id,
            }
        }

        result = self.graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config,
        )

        answer = result["messages"][-1].content

        return {
            "conversation_id": conversation_id,
            "answer": str(answer),
            "standalone_query": result.get(
                "standalone_query",
                question,
            ),
            "sources": result.get(
                "sources",
                [],
            ),
        }

    def close(self) -> None:
        self._database_connection.close()
