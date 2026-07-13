from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb

from src.indexing.embedder import EmbedderConfig, LawEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "law_chunks.json"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "law_chunks")


def _sanitize_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata chỉ chấp nhận str/int/float/bool (không nhận list/dict).

    Vì vậy list/dict phải được làm phẳng thành string trước khi lưu:
    - referenced_articles (list[str])  -> chuỗi phân tách bằng dấu phẩy
    - references (list[dict])          -> JSON string (để phục vụ
      cross-reference resolution ở bước retrieval sau này)
    """
    metadata = dict(chunk["metadata"])

    referenced_articles = metadata.get("referenced_articles") or []
    metadata["referenced_articles"] = ",".join(referenced_articles)

    metadata["references_json"] = json.dumps(
        chunk.get("references", []),
        ensure_ascii=False,
    )

    metadata["law_id"] = metadata.get("law_id") or ""
    metadata["page_start"] = int(metadata.get("page_start") or 0)
    metadata["page_end"] = int(metadata.get("page_end") or 0)

    # Loại bỏ mọi giá trị None còn sót (Chroma raise lỗi với None).
    return {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {CHUNKS_PATH}. "
            "Hãy chạy ingest pipeline (src/ingest/run.py) trước."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build(
    *,
    embedder: LawEmbedder | None = None,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    chunks = load_chunks()
    embedder = embedder or LawEmbedder(EmbedderConfig())

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # recreate collection để lần build sau luôn phản ánh đúng law_chunks.json
    # hiện tại (tránh vector cũ mồ côi khi một Điều bị xoá/sửa ID).
    existing_collections = {col.name for col in client.list_collections()}

    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    embedding_inputs = [chunk["embedding_text"] for chunk in chunks]
    metadatas = [_sanitize_metadata(chunk) for chunk in chunks]

    embeddings = embedder.encode_passages(embedding_inputs)

    batch_size = 64
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return {
        "collection_name": collection_name,
        "persist_dir": str(persist_dir),
        "model_name": embedder.config.model_name,
        "dimension": embedder.dimension,
        "chunk_count": len(chunks),
    }


if __name__ == "__main__":
    stats = build()
    print(json.dumps(stats, ensure_ascii=False, indent=2))