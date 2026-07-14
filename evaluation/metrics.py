from __future__ import annotations

from typing import Any


def document_key(metadata: dict[str, Any]) -> tuple[str, str]:
    return (
        str(metadata.get("law_id", "")).strip().lower(),
        str(metadata.get("article", "")).strip().lower(),
    )


def relevant_keys(question: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (
            str(item["law_id"]).strip().lower(),
            str(item["article"]).strip().lower(),
        )
        for item in question.get("relevant_articles", [])
    }


def recall_at_k(
    retrieved_metadata: list[dict[str, Any]],
    expected: set[tuple[str, str]],
    k: int,
) -> float:
    if not expected:
        return 0.0

    retrieved = {
        document_key(metadata)
        for metadata in retrieved_metadata[:k]
    }

    return len(retrieved & expected) / len(expected)


def hit_at_k(
    retrieved_metadata: list[dict[str, Any]],
    expected: set[tuple[str, str]],
    k: int,
) -> float:
    if not expected:
        return 0.0

    retrieved = {
        document_key(metadata)
        for metadata in retrieved_metadata[:k]
    }

    return float(bool(retrieved & expected))


def reciprocal_rank(
    retrieved_metadata: list[dict[str, Any]],
    expected: set[tuple[str, str]],
) -> float:
    if not expected:
        return 0.0

    for rank, metadata in enumerate(retrieved_metadata, start=1):
        if document_key(metadata) in expected:
            return 1.0 / rank

    return 0.0