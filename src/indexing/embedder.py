from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from sentence_transformers import SentenceTransformer

# Model nên benchmark trước khi chốt (xem evaluation/eval_retrieval.py):
#   - "bkai-foundation-models/vietnamese-bi-encoder"  (baseline tiếng Việt,
#     KHÔNG cần prefix "query:"/"passage:")
#   - "intfloat/multilingual-e5-base"                  (baseline đa ngôn ngữ,
#     BẮT BUỘC prefix "query: " / "passage: ", nếu không recall sẽ giảm rõ rệt)
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"

# Models thuộc họ E5 yêu cầu prefix "query: " / "passage: " để encode đúng
# không gian vector đã được huấn luyện. Thiếu bước này là lỗi phổ biến nhất
# khiến retrieval bị lệch dù model đúng.
_E5_FAMILY_PREFIXES = ("e5-", "multilingual-e5", "intfloat/e5")


@dataclass
class EmbedderConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" nếu có GPU
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    normalize_embeddings: bool = True


class LawEmbedder:
    """Bọc sentence-transformers để encode passage/query nhất quán.

    Dùng cùng một class cho cả bước index (encode_passages) và bước
    retrieval (encode_query) để đảm bảo passage và query luôn nằm trong
    cùng một không gian vector — nếu lệch prefix giữa hai bước, cosine
    similarity sẽ sai một cách âm thầm (không raise lỗi, chỉ recall thấp).
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        self._is_e5 = any(
            marker in self.config.model_name.lower()
            for marker in _E5_FAMILY_PREFIXES
        )

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)

        if self._is_e5:
            texts = [f"passage: {text}" for text in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=len(texts) > 50,
        )

        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        query_text = f"query: {text}" if self._is_e5 else text

        embedding = self._model.encode(
            [query_text],
            normalize_embeddings=self.config.normalize_embeddings,
        )

        return embedding[0].tolist()