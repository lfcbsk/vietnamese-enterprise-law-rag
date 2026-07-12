from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class VietnameseLawParser:
    def __init__(self , law_id: str = "59/2020/QH14",law_name: str = "Luật Doanh nghiệp 2020",) -> None:
        self.law_name = law_name
        self.law_id = law_id

        self.page_pattern = re.compile(
            r"^\[PAGE\s+(\d+)\]$",
            re.IGNORECASE,)

        self.chapter_pattern = re.compile(
            r"^\s*(Chương\s+[IVXLCDM\d]+)"
            r"\s*[.:\-–]?\s*(.*)$",
            re.IGNORECASE,)

        self.article_heading_pattern = re.compile(
            r"^\s*(Điều\s+(\d+[A-Za-z]?))"
            r"\s*[.:\-–]\s*(.*)$",
            re.IGNORECASE,)

        self.article_heading_only_pattern = re.compile(
            r"^\s*(Điều\s+(\d+[A-Za-z]?))"
            r"\s*[.:\-–]?\s*$",
            re.IGNORECASE,)

        # các case điều này dẫn từ điều luật kia
        self.article_reference_pattern = re.compile(
            r"^của\s+Luật\s+này\b",
            re.IGNORECASE,)

        self.reference_pattern = re.compile(r"""
            (?:điểm\s+(?P<point>[a-zđ])\s*)?
            (?:khoản\s+(?P<clause>\d+[a-z]?)\s*)?
            Điều\s+(?P<article>\d+[A-Za-z]?)
            (?:\s+của\s+
                (?P<target_law>
                    Luật\s+này|Luật\s+[^,.;\n]+))?
            """,
            re.IGNORECASE | re.VERBOSE,)


    def _extract_references(
        self,
        text: str,
        current_article: str,) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        current_article_number_match = re.search(
            r"\d+[A-Za-z]?",
            current_article,
        )

        current_article_number = (
            current_article_number_match.group(0)
            if current_article_number_match
            else None
        )

        for match in self.reference_pattern.finditer(text):
            article_number = match.group("article")
            clause_number = match.group("clause")
            point = match.group("point")
            target_law = match.group("target_law")

            # Bỏ trường hợp heading của chính chunk bị nhận thành reference.
            if (
                article_number == current_article_number
                and clause_number is None
                and point is None
                and match.start() == 0
            ):
                continue

            normalized_target_law = (
                target_law.strip()
                if target_law
                else "Luật này"
            )

            key = (
                normalized_target_law.lower(),
                article_number.lower(),
                clause_number.lower() if clause_number else None,
                point.lower() if point else None,
            )

            if key in seen:
                continue

            seen.add(key)

            references.append(
                {
                    "target_law_text": normalized_target_law,
                    "target_law_id": (
                        self.law_id
                        if normalized_target_law.lower() == "luật này"
                        else None
                    ),
                    "target_article": article_number,
                    "target_clause": clause_number,
                    "target_point": point,
                    "reference_text": match.group(0).strip(),
                    "resolution_status": (
                        "resolved_same_law"
                        if normalized_target_law.lower() == "luật này"
                        else "unresolved"
                    ),
                }
            )

        return references

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

            article_match = self.article_heading_pattern.match(line)
            article_only_match = self.article_heading_only_pattern.match(line)

            if article_match:
                flush_current_article()

                current_article = article_match.group(1)
                current_title = article_match.group(3).strip()
                current_page_start = current_page

                heading = current_article

                if current_title:
                    heading += f". {current_title}"

                current_content = [heading]
                continue

            if article_only_match:
                flush_current_article()

                current_article = article_only_match.group(1)
                current_title = ""
                current_page_start = current_page
                current_content = [current_article]
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

    def _create_chunk(self,*,
        chapter: str,
        article: str,
        title: str,
        content_lines: list[str],
        source: str,
        page_start: int,
        page_end: int,) -> dict[str, Any]:
        full_text = "\n".join(content_lines).strip()

        body_lines = (content_lines[1:] if len(content_lines) > 1 else [])

        body_text = "\n".join(body_lines).strip()

        references = self._extract_references(
            text=body_text,
            current_article=article,)

        source_slug = self._slug(Path(source).stem)
        article_slug = self._slug(article)

        embedding_text = "\n".join(
            part
            for part in [
                self.law_name,
                chapter,
                f"{article}. {title}".strip(),
                body_text,
            ]
            if part
        )

        return {
            "id": f"{source_slug}_{article_slug}",
            "law_id": self.law_id,
            "law_name": self.law_name,
            "chapter": chapter,
            "article": article,
            "article_title": title,
            "content": full_text,
            "embedding_text": embedding_text,
            "references": references,
            "metadata": {
                "source": source,
                "type": "article",
                "law_id": self.law_id,
                "law_name": self.law_name,
                "chapter": chapter,
                "article": article,
                "article_title": title,
                "page_start": page_start,
                "page_end": page_end,
                "referenced_articles": [
                    reference["target_article"]
                    for reference in references
                ],
            },
        }

    @staticmethod
    def _slug(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"\W+", "_", value)

        return value.strip("_")
