from __future__ import annotations

from src.retrieval.schema import RetrievalResult


def build_context(
    results: list[RetrievalResult],
) -> str:
    sections: list[str] = []

    for result in results:
        metadata = result.metadata

        article = metadata.get(
            "article",
            "Không rõ Điều",
        )
        title = metadata.get(
            "article_title",
            "",
        )
        source = metadata.get(
            "source",
            "",
        )
        page_start = metadata.get(
            "page_start",
            "",
        )
        page_end = metadata.get(
            "page_end",
            "",
        )

        citation = (
            f"{article}. {title} | "
            f"Nguồn: {source} | "
            f"Trang: {page_start}-{page_end}"
        )

        sections.append(
            f"[{citation}]\n{result.content}"
        )

    return "\n\n".join(sections)


def serialize_sources(
    results: list[RetrievalResult],
) -> list[dict]:
    return [
        {
            "chunk_id": result.chunk_id,
            "article": result.metadata.get(
                "article",
                "",
            ),
            "article_title": result.metadata.get(
                "article_title",
                "",
            ),
            "source": result.metadata.get(
                "source",
                "",
            ),
            "page_start": result.metadata.get(
                "page_start",
                0,
            ),
            "page_end": result.metadata.get(
                "page_end",
                0,
            ),
            "score": result.score,
        }
        for result in results
    ]