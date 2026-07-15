from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_keys,
)
from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"


def load_questions() -> list[dict[str, Any]]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_retriever(
    name: str,
    *,
    dense_weight: float = 0.9,
    bm25_weight: float = 0.1,
) -> Any:
    if name == "bm25":
        return BM25Retriever()

    if name == "dense":
        return DenseRetriever()

    if name == "hybrid":
        return HybridRetriever(
            source_weights={
                "dense": dense_weight,
                "bm25": bm25_weight,
            }
        )

    raise ValueError(f"Retriever không hợp lệ: {name}")


def evaluate(
    retriever_name: str,
    top_k: int = 5,
    candidate_k: int = 20,
    dense_weight: float = 0.9,
    bm25_weight: float = 0.1,
) -> dict[str, Any]:
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("Trọng số retrieval phải lớn hơn hoặc bằng 0")
    if dense_weight == 0 and bm25_weight == 0:
        raise ValueError("Phải có ít nhất một trọng số retrieval lớn hơn 0")

    questions = load_questions()
    retriever = create_retriever(
        retriever_name,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    details: list[dict[str, Any]] = []

    for question in questions:
        expected = relevant_keys(question)

        # Câu hỏi ngoài phạm vi cần đánh giá riêng,
        # chưa đưa vào retrieval recall/MRR.
        if not expected:
            continue

        if retriever_name == "hybrid":
            results = retriever.search(
                question["question"],
                top_k=top_k,
                candidate_k=candidate_k,
            )
        else:
            results = retriever.search(
                question["question"],
                top_k=top_k,
            )

        retrieved_metadata = [
            result.metadata
            for result in results
        ]

        item = {
            "id": question["id"],
            "question": question["question"],
            "category": question.get("category"),
            "expected": [
                {
                    "law_id": law_id,
                    "article": article,
                }
                for law_id, article in sorted(expected)
            ],
            "retrieved": [
                {
                    "rank": result.rank,
                    "chunk_id": result.chunk_id,
                    "article": result.metadata.get("article"),
                    "article_title": result.metadata.get("article_title"),
                    "score": result.score,
                    "source": result.source,
                }
                for result in results
            ],
            "hit_at_1": hit_at_k(retrieved_metadata, expected, 1),
            "hit_at_3": hit_at_k(retrieved_metadata, expected, 3),
            "hit_at_5": hit_at_k(retrieved_metadata, expected, 5),
            "recall_at_5": recall_at_k(
                retrieved_metadata,
                expected,
                5,
            ),
            "reciprocal_rank": reciprocal_rank(
                retrieved_metadata,
                expected,
            ),
        }

        details.append(item)

    summary = {
        "retriever": retriever_name,
        "question_count": len(details),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "hit_at_1": mean(item["hit_at_1"] for item in details),
        "hit_at_3": mean(item["hit_at_3"] for item in details),
        "hit_at_5": mean(item["hit_at_5"] for item in details),
        "recall_at_5": mean(item["recall_at_5"] for item in details),
        "mrr": mean(item["reciprocal_rank"] for item in details),
    }

    if retriever_name == "hybrid":
        summary["dense_weight"] = dense_weight
        summary["bm25_weight"] = bm25_weight

    return {
        "summary": summary,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--dense-weight", type=float, default=0.9)
    parser.add_argument("--bm25-weight", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    report = evaluate(
        retriever_name=args.retriever,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT_DIR /
        f"retrieval_{args.retriever}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Đã lưu kết quả tại: {output_path}")


if __name__ == "__main__":
    main()
