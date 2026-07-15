"""Kiểm tra và chỉ build lại retrieval indexes khi dữ liệu chưa sẵn sàng."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb

from src.indexing import build_indexes
from src.indexing.build_bm25_index import BM25_INDEX_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "law_chunks.json"
MANIFEST_PATH = DATA_DIR / "indexes" / "manifest.json"
CHROMA_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "law_chunks")
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-base",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def indexes_are_ready() -> bool:
    """Trả về True khi BM25, Chroma và manifest khớp chunks/config hiện tại."""
    required_paths = (
        CHUNKS_PATH,
        BM25_INDEX_PATH,
        MANIFEST_PATH,
        CHROMA_DIR,
    )
    if not all(path.exists() for path in required_paths):
        return False

    try:
        chunks = _load_json(CHUNKS_PATH)
        manifest = _load_json(MANIFEST_PATH)
        dense_manifest = manifest["dense_index"]
        bm25_manifest = manifest["bm25_index"]

        expected_count = len(chunks)
        if expected_count == 0:
            return False
        if dense_manifest.get("chunk_count") != expected_count:
            return False
        if bm25_manifest.get("doc_count") != expected_count:
            return False
        if dense_manifest.get("model_name") != MODEL_NAME:
            return False
        if dense_manifest.get("collection_name") != COLLECTION_NAME:
            return False

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() == expected_count
    except (KeyError, OSError, TypeError, ValueError):
        return False
    except Exception:
        # Chroma dùng các exception khác nhau giữa các phiên bản khi collection
        # chưa tồn tại hoặc database hỏng. Trường hợp đó phải build lại.
        return False


def main() -> None:
    if indexes_are_ready():
        print("Retrieval indexes are ready; skipping rebuild.")
        return

    print("Retrieval indexes are missing or stale; rebuilding...")
    build_indexes.run()


if __name__ == "__main__":
    main()
