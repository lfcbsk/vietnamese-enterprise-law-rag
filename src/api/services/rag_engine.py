from __future__ import annotations

import sqlite3
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
    QUERY_REWRITE_PROMPT,
)
from src.memory.checkpointer import (
    create_sqlite_checkpointer,
)
from src.retrieval import HybridRetriever
from src.retrieval.title_reranker import rerank_by_title


class RAGState(MessagesState):
    standalone_query: str
    context: str
    sources: list[dict[str, Any]]


class RAGEngine:
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        retriever: HybridRetriever,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings

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
            "rewrite_query",
            self._rewrite_query,
        )
        builder.add_node(
            "retrieve",
            self._retrieve,
        )
        builder.add_node(
            "generate",
            self._generate,
        )

        builder.add_edge(START, "rewrite_query")
        builder.add_edge(
            "rewrite_query",
            "retrieve",
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile(
            checkpointer=self.checkpointer
        )

    def _rewrite_query(
        self,
        state: RAGState,
    ) -> dict[str, str]:
        messages = state["messages"]

        human_messages = [
            message
            for message in messages
            if isinstance(message, HumanMessage)
        ]

        current_question = (
            human_messages[-1].content
        )

        # Câu hỏi đầu tiên đã độc lập, không cần gọi LLM
        # thêm một lần để rewrite.
        if len(human_messages) == 1:
            return {
                "standalone_query": str(
                    current_question
                )
            }

        recent_messages = messages[-8:]

        history = "\n".join(
            f"{message.type}: {message.content}"
            for message in recent_messages
        )

        rewrite_message = HumanMessage(
            content=QUERY_REWRITE_PROMPT.format(
                history=history
            )
        )

        response = self.llm.invoke(
            [rewrite_message]
        )

        return {
            "standalone_query": str(
                response.content
            ).strip()
        }

    def _retrieve(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        candidates = self.retriever.search(
            state["standalone_query"],
            top_k=self.settings.rag_candidate_k,      # lấy nhiều ứng viên, vd 40
            candidate_k=self.settings.rag_candidate_k,
        )
        results = rerank_by_title(
            state["standalone_query"], candidates, top_k=self.settings.rag_top_k
        )

        return {
            "context": build_context(results),
            "sources": serialize_sources(
                results
            ),
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
        recent_messages = state["messages"][-8:]

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