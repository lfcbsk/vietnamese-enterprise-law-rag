from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.indexing.tokenize import vietnamese_tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "law_chunks.json"
INDEX_DIR = DATA_DIR / "indexes"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {CHUNKS_PATH}. "
            "Hãy chạy ingest pipeline (src/ingest/run.py) trước."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build() -> dict[str, Any]:
    chunks = load_chunks()

    # BM25 nên đánh index trên `embedding_text` (đã ghép tên luật + Chương +
    # Điều + nội dung) để câu hỏi chứa "Luật Doanh nghiệp" hay tên Chương
    # cũng match được, không chỉ nội dung Khoản.
    tokenized_corpus = [
        vietnamese_tokenize(chunk["embedding_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    doc_ids = [chunk["id"] for chunk in chunks]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with BM25_INDEX_PATH.open("wb") as file:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, file)

    avg_tokens = (
        sum(len(tokens) for tokens in tokenized_corpus) / len(tokenized_corpus)
        if tokenized_corpus
        else 0
    )

    return {
        "index_path": str(BM25_INDEX_PATH),
        "doc_count": len(doc_ids),
        "avg_tokens_per_doc": round(avg_tokens, 1),
    }


def load_index() -> tuple[BM25Okapi, list[str]]:
    """Dùng ở bước retrieval: trả về (bm25, doc_ids) đã build sẵn."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {BM25_INDEX_PATH}. Hãy chạy build() trước."
        )

    with BM25_INDEX_PATH.open("rb") as file:
        payload = pickle.load(file)

    return payload["bm25"], payload["doc_ids"]


def search(question: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Trả về [(chunk_id, bm25_score), ...] sắp xếp giảm dần theo score."""
    bm25, doc_ids = load_index()
    tokenized_query = vietnamese_tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(doc_ids, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return ranked[:top_k]


if __name__ == "__main__":
    stats = build()
    print(json.dumps(stats, ensure_ascii=False, indent=2))