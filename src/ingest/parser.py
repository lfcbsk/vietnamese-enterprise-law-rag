from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class VietnameseLawParser:
    ARTICLE_CHAPTER_RANGES = (
        (1, 16, "Chương I"),
        (17, 73, "Chương II"),
        (74, 110, "Chương III"),
        (111, 176, "Chương IV"),
        (177, 187, "Chương V"),
        (188, 197, "Chương VI"),
        (198, 205, "Chương VII"),
        (206, 210, "Chương VIII"),
        (211, 216, "Chương IX"),
        (217, 218, "Chương X"),
    )

    def __init__(
        self,
        law_id: str = "59/2020/QH14",
        law_name: str = "Luật Doanh nghiệp 2020",
    ) -> None:
        self.law_name = law_name
        self.law_id = law_id

        self.page_pattern = re.compile(
            r"^\[PAGE\s+(\d+)\]$",
            re.IGNORECASE,)

        self.chapter_pattern = re.compile(
            r"^\s*(?:[|“”\"'`._–—-]\s*){0,4}"
                r"(Chương\s+[IVXLCDM\d]+)"
                r"\s*[.:\-–—]?\s*(.*)$",
                re.IGNORECASE,)

        self.article_heading_pattern = re.compile(
            r"^\s*(?:[|“”\"'`._–—-]\s*){0,4}"
            r"(Điều\s+(\d+[A-Za-z]?))"
            r"\s*[.:\-–—]\s*(.*)$",
            re.IGNORECASE,)

        self.article_heading_only_pattern = re.compile(
            r"^\s*(?:[|“”\"'`._–—-]\s*){0,4}"
            r"(Điều\s+(\d+[A-Za-z]?))"
            r"\s*[.:\-–—]?\s*$",
            re.IGNORECASE,)

        # OCR có thể chèn một cụm rác trước heading, ví dụ:
        # "viên Điều 61.", "ngườiĐiều 102.", "ln Điều 201.".
        # Pattern này chỉ được chấp nhận khi số Điều đúng bằng Điều kế tiếp.
        self.noisy_article_heading_pattern = re.compile(
            r"^.{0,30}?Điều\s+(\d+[A-Za-z]?)"
            r"\s*[.:\-–—]+\s*(.+)$",
            re.IGNORECASE,
        )

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
        
        self.embedded_article_heading_pattern = re.compile(
            r"^.{0,30}?Điều\s+(\d+[A-Za-z]?)"
            r"\s*[.:\-–—]",
            re.IGNORECASE | re.MULTILINE,
        )

    @staticmethod
    def _extract_article_number(article: str) -> int | None:
        match = re.search(r"\d+", article)
        return int(match.group()) if match else None

    @classmethod
    def _resolve_chapter(
        cls,
        article: str,
        detected_chapter: str,
    ) -> str:
        article_number = cls._extract_article_number(article)

        if article_number is None:
            return detected_chapter

        for start, end, chapter in cls.ARTICLE_CHAPTER_RANGES:
            if start <= article_number <= end:
                return chapter

        return detected_chapter

    def _match_article_heading(
        self,
        line: str,
        current_article: str,
        previous_article_number: int | None = None,
    ) -> tuple[str, str] | None:
        regular_match = self.article_heading_pattern.match(line)

        if regular_match:
            return (
                regular_match.group(1),
                regular_match.group(3).strip(),
            )

        only_match = self.article_heading_only_pattern.match(line)

        if only_match:
            return only_match.group(1), ""

        noisy_match = self.noisy_article_heading_pattern.match(line)

        if not noisy_match:
            return None

        detected_number = int(noisy_match.group(1))
        current_number = (
            self._extract_article_number(current_article)
            if current_article
            else previous_article_number
        )

        # Pattern dự phòng khá rộng, nên chỉ tin khi đây chính xác là
        # Điều tiếp theo. Dẫn chiếu "Điều X của Luật này" sẽ bị loại.
        if (
            current_number is None
            or detected_number != current_number + 1
        ):
            return None

        title = re.sub(
            r"\s*[|¦`_]+\s*$",
            "",
            noisy_match.group(2),
        ).strip()

        return f"Điều {detected_number}", title


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

    def validate_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        errors: list[str] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            chunk_id = chunk["id"]

            if chunk_id in seen_ids:
                errors.append(f"ID trùng: {chunk_id}")

            seen_ids.add(chunk_id)

            if not chunk["content"].strip():
                errors.append(f"Chunk rỗng: {chunk_id}")
                continue

            expected_match = re.search(
                r"\d+[A-Za-z]?",
                chunk["article"],
            )
            expected_article = (
                expected_match.group(0).lower()
                if expected_match
                else None
            )

            detected_articles = [
                number.lower()
                for number
                in self.embedded_article_heading_pattern.findall(
                    chunk["content"]
                )
            ]

            unexpected_articles = [
                number
                for number in detected_articles
                if number != expected_article
            ]

            if unexpected_articles:
                errors.append(
                    f"{chunk_id} chứa heading Điều khác: "
                    f"{unexpected_articles}"
                )

        if errors:
            error_preview = "\n".join(errors[:20])

            raise ValueError(
                "Kết quả parse không hợp lệ:\n"
                f"{error_preview}"
            )
    def parse(self,text: str,source: str,) -> list[dict[str, Any]]:

        chunks: list[dict[str, Any]] = []

        current_page = 1
        current_page_start = 1
        current_content_page_end = 1

        current_chapter = "Phần mở đầu"
        current_article = ""
        current_title = ""
        current_content: list[str] = []
        previous_article_number: int | None = None

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
                page_end=current_content_page_end,
            )

            chunks.append(chunk)
            current_content = []

        for raw_line in text.splitlines():
            line = self._normalize_line_for_parsing(raw_line)

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

            article_heading = self._match_article_heading(
                line=line,
                current_article=current_article,
                previous_article_number=previous_article_number,
            )

            if article_heading:
                flush_current_article()

                current_article, current_title = article_heading
                previous_article_number = (
                    self._extract_article_number(current_article)
                )
                current_page_start = current_page
                current_content_page_end = current_page

                heading = current_article

                if current_title:
                    heading = f"{current_article}. {current_title}"

                current_content = [heading]
                continue

            if current_article:
                is_clause = bool(
                    re.match(r"^\d+[.)]\s+", line)
                )
                is_point = bool(
                    re.match(
                        r"^[a-zđ][.)]\s+",
                        line,
                        re.IGNORECASE,
                    )
                )

                if (
                    not current_title
                    and len(current_content) == 1
                    and not is_clause
                    and not is_point
                ):
                    current_title = line
                    current_content[0] = (
                        f"{current_article}. {current_title}"
                    )
                    current_content_page_end = current_page
                    continue

                current_content.append(line)
                current_content_page_end = current_page

        flush_current_article()
        self.validate_chunks(chunks)

        return chunks

    def _create_chunk(self, *,
        chapter: str,
        article: str,
        title: str,
        content_lines: list[str],
        source: str,
        page_start: int,
        page_end: int,) -> dict[str, Any]:
        resolved_chapter = self._resolve_chapter(
            article=article,
            detected_chapter=chapter,
        )

        full_text = "\n".join(content_lines).strip()

        body_lines = (content_lines[1:] if len(content_lines) > 1 else [])

        body_text = "\n".join(body_lines).strip()

        references = self._extract_references(
            text=body_text,
            current_article=article,)

        source_slug = self._slug(Path(source).stem)
        article_slug = self._slug(article)

        article_heading = article

        if title:
            article_heading = f"{article}. {title}"

        embedding_text = "\n".join(
            part
            for part in [
                self.law_name,
                resolved_chapter,
                article_heading,
                body_text,
            ]
            if part
        )

        return {
            "id": f"{source_slug}_{article_slug}",
            "law_id": self.law_id,
            "law_name": self.law_name,
            "chapter": resolved_chapter,
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
                "chapter": resolved_chapter,
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
    @staticmethod
    def _normalize_line_for_parsing(raw_line: str) -> str:
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            return ""

        # Chỉ xóa rác OCR nếu ngay sau đó là heading Chương/Điều.
        line = re.sub(
            r"^\s*(?:[|“”\"'`._–—-]\s*){1,4}"
            r"(?=(?:Điều\s+\d+|Chương\s+[IVXLCDM\d]+)\b)",
            "",
            line,
            flags=re.IGNORECASE,
        )

        # Xóa ký tự bảng ở cuối heading.
        if re.match(
            r"^(?:Điều\s+\d+|Chương\s+[IVXLCDM\d]+)\b",
            line,
            re.IGNORECASE,
        ):
            line = re.sub(r"\s*[|`]+$", "", line).strip()

        return line