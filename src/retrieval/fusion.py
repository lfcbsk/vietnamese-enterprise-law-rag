from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from src.retrieval.schema import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    *,
    top_k: int = 10,
    rrf_k: int = 60,
    source_weights: dict[str, float] | None = None,
) -> list[RetrievalResult]:
    """Gộp nhiều ranking bằng RRF, có hỗ trợ trọng số theo nguồn."""
    if top_k <= 0:
        return []
    if rrf_k < 0:
        raise ValueError("rrf_k phải lớn hơn hoặc bằng 0")

    weights = source_weights or {}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Trọng số retrieval phải lớn hơn hoặc bằng 0")

    fusion_scores: dict[str, float] = defaultdict(float)
    result_by_id: dict[str, RetrievalResult] = {}
    component_scores: dict[str, dict[str, float]] = defaultdict(dict)
    component_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for results in result_lists:
        seen_in_ranking: set[str] = set()

        for fallback_rank, result in enumerate(results, start=1):
            if result.chunk_id in seen_in_ranking:
                continue

            seen_in_ranking.add(result.chunk_id)
            rank = result.rank if result.rank > 0 else fallback_rank
            weight = float(weights.get(result.source, 1.0))

            fusion_scores[result.chunk_id] += weight / (rrf_k + rank)
            result_by_id.setdefault(result.chunk_id, result)
            component_scores[result.chunk_id][result.source] = result.score
            component_ranks[result.chunk_id][result.source] = rank

    ranked_ids = sorted(
        fusion_scores,
        key=lambda chunk_id: (-fusion_scores[chunk_id], chunk_id),
    )[:top_k]

    fused: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        original = result_by_id[chunk_id]
        metadata = dict(original.metadata)
        metadata["retrieval_scores"] = component_scores[chunk_id]
        metadata["retrieval_ranks"] = component_ranks[chunk_id]

        fused.append(
            RetrievalResult(
                chunk_id=chunk_id,
                content=original.content,
                score=fusion_scores[chunk_id],
                rank=rank,
                source="hybrid",
                metadata=metadata,
            )
        )

    return fused
