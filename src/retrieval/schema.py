from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Một chunk được trả về bởi một retriever.

    ``score`` chỉ có ý nghĩa trong phạm vi của ``source``. BM25 score và
    cosine similarity không cùng thang đo; bước hybrid phải fusion theo rank.
    """

    chunk_id: str
    content: str
    score: float
    rank: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
