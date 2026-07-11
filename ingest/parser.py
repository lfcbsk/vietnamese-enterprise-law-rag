from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class VietnameseLawParser:
    def __init__(
        self,
        law_name: str = "Luật Doanh nghiệp 2020",
    ) -> None:
        self.law_name = law_name

        self.page_pattern = re.compile(
            r"^\[PAGE\s+(\d+)\]$",
            re.IGNORECASE,
        )

        self.chapter_pattern = re.compile(
            r"^\s*(Chương\s+[IVXLCDM\d]+)"
            r"\s*[.:\-–]?\s*(.*)$",
            re.IGNORECASE,
        )

        self.article_pattern = re.compile(
            r"^\s*(Điều\s+\d+[A-Za-z]?)"
            r"(?![\w⁰¹²³⁴⁵⁶⁷⁸⁹])"
            r"\s*[.:\-–]?\s*(.*)$",
            re.IGNORECASE,
        )

        # Dòng nội dung có thể bắt đầu bằng một dẫn chiếu, ví dụ
        # "Điều 88 của Luật này...". Đây không phải tiêu đề Điều mới.
        self.article_reference_pattern = re.compile(
            r"^của\s+Luật\s+này\b",
            re.IGNORECASE,
        )

    def parse(
        self,
        text: str,
        source: str,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        current_page = 1
        current_page_start = 1
        current_chapter = "Phần mở đầu"
        current_article = ""
        current_title = ""
        current_content: list[str] = []

        def flush_current_article() -> None:
            nonlocal current_content

            if not current_article:
                return

            chunk = self._create_chunk(
                chapter=current_chapter,
                article=current_article,
                title=current_title,
                content_lines=current_content,
                source=source,
                page_start=current_page_start,
                page_end=current_page,
            )

            chunks.append(chunk)
            current_content = []

        for raw_line in text.splitlines():
            line = re.sub(
                r"\s+",
                " ",
                raw_line,
            ).strip()

            if not line:
                continue

            page_match = self.page_pattern.match(line)

            if page_match:
                current_page = int(page_match.group(1))
                continue

            chapter_match = self.chapter_pattern.match(line)

            if chapter_match:
                flush_current_article()

                current_chapter = chapter_match.group(1)
                current_article = ""
                current_title = ""
                current_content = []

                continue

            article_match = self.article_pattern.match(line)

            if article_match:
                possible_title = article_match.group(2).strip()

                if self.article_reference_pattern.match(possible_title):
                    if current_article:
                        current_content.append(line)
                    continue

                flush_current_article()

                current_article = article_match.group(1)
                current_title = possible_title
                current_page_start = current_page

                heading = current_article

                if current_title:
                    heading += f". {current_title}"

                current_content = [heading]

                continue

            if current_article:
                # OCR đôi khi để tiêu đề Điều ở dòng tiếp theo.
                if (
                    not current_title
                    and len(current_content) == 1
                ):
                    current_title = line

                current_content.append(line)

        flush_current_article()

        return chunks

    def _create_chunk(
        self,
        *,
        chapter: str,
        article: str,
        title: str,
        content_lines: list[str],
        source: str,
        page_start: int,
        page_end: int,
    ) -> dict[str, Any]:
        full_text = "\n".join(content_lines).strip()

        source_slug = self._slug(Path(source).stem)
        article_slug = self._slug(article)

        embedding_text = (
            f"{self.law_name}\n"
            f"{chapter}\n"
            f"{article}. {title}\n"
            f"{full_text}"
        )

        return {
            "id": f"{source_slug}_{article_slug}",
            "law_name": self.law_name,
            "chapter": chapter,
            "article": article,
            "article_title": title,
            "content": full_text,
            "embedding_text": embedding_text,
            "metadata": {
                "source": source,
                "type": "article",
                "chapter": chapter,
                "article": article,
                "article_title": title,
                "page_start": page_start,
                "page_end": page_end,
            },
        }

    @staticmethod
    def _slug(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"\W+", "_", value)

        return value.strip("_")
