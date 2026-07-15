from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.retrieval.schema import RetrievalResult


DEFAULT_CHUNKS_PATH = Path("data/law_chunks.json")
_ARTICLE_REFERENCE = re.compile(
    r"\bdieu\s+(\d+[a-z]?)\b",
    re.IGNORECASE,
)


def _fold_text(text: str) -> str:
    """Bỏ dấu tiếng Việt để nhận cả ``Điều 111`` và ``dieu 111``."""
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def extract_article_numbers(query: str) -> list[str]:
    """Lấy các số điều được nhắc tường minh, giữ nguyên thứ tự và bỏ trùng."""
    matches = _ARTICLE_REFERENCE.findall(_fold_text(query))
    return list(dict.fromkeys(match.lower() for match in matches))


def _article_number(article: str) -> str | None:
    match = re.search(r"(\d+[a-z]?)", _fold_text(article), re.IGNORECASE)
    return match.group(1).lower() if match else None


class ArticleLookup:
    """Tra cứu xác định theo số điều từ dữ liệu ingestion gốc."""

    def __init__(self, chunks_path: Path | str = DEFAULT_CHUNKS_PATH) -> None:
        self.chunks_path = Path(chunks_path)
        self._articles = self._load_articles()

    def _load_articles(self) -> dict[str, list[dict[str, Any]]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu điều luật: {self.chunks_path}. "
                "Hãy chạy `python -m src.ingest.run` trước."
            )

        with self.chunks_path.open(encoding="utf-8") as file:
            chunks = json.load(file)

        articles: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            number = _article_number(str(chunk.get("article", "")))
            if number is not None:
                articles.setdefault(number, []).append(chunk)
        return articles

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Trả đúng chunk khi query nêu ``Điều N``; query khác trả danh sách rỗng."""
        if top_k <= 0:
            return []

        requested_numbers = extract_article_numbers(query)
        if not requested_numbers:
            return []

        matched_chunks = [
            chunk
            for number in requested_numbers
            for chunk in self._articles.get(number, [])
        ]

        results: list[RetrievalResult] = []
        for rank, chunk in enumerate(matched_chunks[:top_k], start=1):
            metadata = dict(chunk.get("metadata", {}))
            for field in (
                "law_id",
                "law_name",
                "chapter",
                "article",
                "article_title",
            ):
                if field not in metadata and field in chunk:
                    metadata[field] = chunk[field]

            results.append(
                RetrievalResult(
                    chunk_id=str(chunk["id"]),
                    content=str(chunk["content"]),
                    score=1.0,
                    rank=rank,
                    source="article_lookup",
                    metadata=metadata,
                )
            )
        return results
