from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from src.retrieval.schema import RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "law_chunks")


class QueryEmbedder(Protocol):
    def encode_query(self, text: str) -> list[float]: ...


class DenseRetriever:
    """Semantic retriever đọc vector từ ChromaDB.

    Có thể truyền ``collection`` và ``embedder`` giả trong unit test. Khi không
    truyền, class dùng đúng cấu hình mặc định của bước build dense index.
    """

    def __init__(
        self,
        *,
        embedder: QueryEmbedder | None = None,
        collection: Any | None = None,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedder = embedder or self._create_default_embedder()

        if collection is not None:
            self.collection = collection
            return

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài chromadb. Hãy chạy `python -m pip install -e .` "
                "hoặc `python -m pip install chromadb`."
            ) from exc

        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dense index tại {persist_dir}. "
                "Hãy chạy `python -m src.indexing.build_dense_index` trước."
            )

        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"Không tìm thấy Chroma collection '{collection_name}' "
                f"tại {persist_dir}. Hãy build lại dense index."
            ) from exc

    @staticmethod
    def _create_default_embedder() -> QueryEmbedder:
        try:
            from src.indexing.embedder import EmbedderConfig, LawEmbedder
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài sentence-transformers. Hãy chạy "
                "`python -m pip install -e .`."
            ) from exc

        return LawEmbedder(EmbedderConfig())

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query = query.strip()
        if not query or top_k <= 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [self.embedder.encode_query(query)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        response = self.collection.query(**query_kwargs)
        ids = self._first_row(response.get("ids"))
        documents = self._first_row(response.get("documents"))
        metadatas = self._first_row(response.get("metadatas"))
        distances = self._first_row(response.get("distances"))

        results: list[RetrievalResult] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    content=(
                        str(documents[index])
                        if index < len(documents) and documents[index] is not None
                        else ""
                    ),
                    score=1.0 - distance,
                    rank=index + 1,
                    source="dense",
                    metadata=(
                        dict(metadatas[index])
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return self.search(query, top_k=top_k, where=where)

    @staticmethod
    def _first_row(value: Any) -> list[Any]:
        if not value or not isinstance(value, (list, tuple)):
            return []
        first = value[0]
        return list(first) if isinstance(first, (list, tuple)) else []
