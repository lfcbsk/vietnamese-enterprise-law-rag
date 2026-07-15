from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pymupdf


class PDFLoader:
    def __init__(
        self,
        file_path: str | Path,
        *,
        language: str = "vie+eng",
        ocr_dpi: int = 300,
        min_text_length: int = 80,
        tessdata_path: str | Path | None = None,
        use_cache: bool = True,
        force_ocr: bool = False,
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.language = language
        self.ocr_dpi = ocr_dpi
        self.min_text_length = min_text_length
        self.use_cache = use_cache
        self.force_ocr = force_ocr

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy PDF: {self.file_path}"
            )

        self.tessdata_path = self._resolve_tessdata_path(
            tessdata_path
        )

        self.cache_path = (
            self.file_path.parent
            / "ocr"
            / f"{self.file_path.stem}.txt"
        )

        self._validate_languages()

    def load_and_clean(self) -> str:
        if (
            self.use_cache
            and not self.force_ocr
            and self.cache_path.exists()
        ):
            print(f"Using OCR cache: {self.cache_path}")

            return self.clean_text(
                self.cache_path.read_text(encoding="utf-8")
            )

        document = pymupdf.open(self.file_path)
        page_texts: list[str] = []

        try:
            for page_index, page in enumerate(document):
                page_number = page_index + 1

                native_text = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                use_native_text = (
                    not self.force_ocr
                    and self._is_usable_native_text(
                        native_text,
                        min_text_length=self.min_text_length,
                    )
                )

                if use_native_text:
                    text = native_text

                    print(
                        f"Page {page_number}: "
                        f"native text ({len(text)} chars)"
                    )
                else:
                    print(
                        f"Page {page_number}: "
                        "native text không đạt chất lượng, đang OCR..."
                    )

                    text_page = page.get_textpage_ocr(
                        language=self.language,
                        dpi=self.ocr_dpi,
                        full=True,
                        tessdata=str(self.tessdata_path),
                    )

                    text = page.get_text(
                        "text",
                        textpage=text_page,
                        sort=True,
                    ).strip()

                    print(
                        f"Page {page_number}: "
                        f"OCR output ({len(text)} chars)"
                    )

                text = self.clean_text(text)

                # Giữ số trang cho citation sau này.
                page_texts.append(
                    f"[PAGE {page_number}]\n{text}"
                )

        finally:
            document.close()

        full_text = "\n\n".join(page_texts).strip()

        if not full_text:
            raise RuntimeError(
                f"Không lấy được text từ {self.file_path.name}"
            )

        quality = self.calculate_text_quality(full_text)
        print(
            "Text quality: "
            f"{quality['article_headings']} article headings, "
            f"{quality['chapter_headings']} chapter headings, "
            "suspicious ratio="
            f"{quality['suspicious_ratio']:.4%}"
        )

        if self.use_cache:
            self.cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.cache_path.write_text(
                full_text,
                encoding="utf-8",
            )

            print(f"Saved OCR cache: {self.cache_path}")

        return full_text

    @staticmethod
    def _is_usable_native_text(
        text: str,
        *,
        min_text_length: int,
    ) -> bool:
        """Kiểm tra native text có đại diện cho nội dung luật hay không.

        Chỉ kiểm tra độ dài là không đủ: trang đầu của PDF có một lớp
        native text ngắn chứa thông tin chữ ký điện tử, trong khi nội dung
        Điều 1--4 nằm trong ảnh. Trường hợp đó phải chuyển sang OCR.
        """
        if len(text.strip()) < min_text_length:
            return False

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

        legal_markers = (
            "điều ",
            "chương ",
            "khoản ",
            "luật ",
            "doanh nghiệp",
        )
        signature_markers = (
            "ký bởi:",
            "email:",
            "thời gian ký:",
            "cơ quan:",
        )

        legal_marker_count = sum(
            marker in normalized
            for marker in legal_markers
        )
        signature_marker_count = sum(
            marker in normalized
            for marker in signature_markers
        )

        # Lớp text chỉ chứa chữ ký/metadata không đại diện cho ảnh trang.
        if (
            signature_marker_count >= 2
            and legal_marker_count == 0
        ):
            return False

        # Với corpus pháp luật, native text phải có ít nhất một dấu hiệu
        # nội dung pháp lý. Nếu không, OCR toàn trang an toàn hơn.
        return legal_marker_count > 0

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\ufeff", "")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")

        # Trong các PDF hiện tại, glyph "o" thường bị trích xuất/OCR
        # nhầm thành ký tự masculine ordinal "º" (dºanh, hºặc, theº...).
        # Ký tự này không thuộc chính tả tiếng Việt nên có thể thay an toàn.
        text = text.replace("º", "o")

        # OCR thường nhận nhầm chữ số trong số Điều thành ký tự mũ,
        # ví dụ: "Điều 1²0" thay vì "Điều 120".
        text = text.translate(
            str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        )

        # Một số lỗi thường gặp ở tiêu đề pháp luật.
        text = re.sub(
            r"(?im)^\s*điêu(?=\s+\d+)",
            "Điều",
            text,
        )
        text = re.sub(
            r"(?im)^\s*diều(?=\s+\d+)",
            "Điều",
            text,
        )

        # Sửa các heading bị OCR phá vỡ nghiêm trọng trong hai PDF của
        # Luật Doanh nghiệp 2020. Đây là sửa chữa có mục tiêu, không áp
        # dụng như quy tắc tổng quát cho văn bản luật khác.
        text = re.sub(
            r"(?im)^.*Điềuviên79\..*$",
            (
                "Điều 79. Cơ cấu tổ chức quản lý của công ty "
                "trách nhiệm hữu hạn một thành viên do tổ chức "
                "làm chủ sở hữu"
            ),
            text,
        )
        text = re.sub(
            r"(?im)^Điều3\..*?105\.\s*Quyền của Ban kiểm soát.*$",
            "Điều 105. Quyền của Ban kiểm soát",
            text,
        )
        text = re.sub(
            r"(?im)^.*Di[eé]u205\.\s*Chuyển đổi doanh nghiệp tư nhân.*$",
            (
                "Điều 205. Chuyển đổi doanh nghiệp tư nhân thành "
                "công ty trách nhiệm hữu hạn, công ty cổ phần, "
                "công ty hợp danh"
            ),
            text,
        )
        text = re.sub(
            r"(?im)^.*CONGTY\s+CO\s+PHAN\s*$",
            "Điều 111. Công ty cổ phần",
            text,
        )

        text = PDFLoader._fix_joined_words(text)
        text = PDFLoader._fix_missing_diacritics(text)

        # Chỉ bỏ ký tự kẻ bảng/nhiễu khi chúng nằm riêng hoặc ở rìa dòng.
        # Không xóa toàn cục để tránh làm thay đổi nội dung pháp lý hợp lệ.
        text = re.sub(r"(?m)^\s*[|¦`]+\s*$", "", text)
        text = re.sub(r"(?m)^\s*[|¦`]+\s*", "", text)
        text = re.sub(r"(?m)\s*[|¦`]+\s*$", "", text)

        # Giữ xuống dòng vì parser dùng cấu trúc dòng.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def calculate_text_quality(
        text: str,
    ) -> dict[str, float | int]:
        """Trả về các chỉ báo đơn giản để audit chất lượng OCR."""
        text_length = max(len(text), 1)
        suspicious_characters = {
            "pipe": text.count("|"),
            "backtick": text.count("`"),
            "broken_bar": text.count("¦"),
            "capital_i_circumflex": text.count("Î"),
            "replacement": text.count("�"),
        }
        suspicious_total = sum(suspicious_characters.values())

        article_headings = len(
            re.findall(
                r"(?im)^.{0,30}?Điều\s+\d+\s*[.:\-–—]",
                text,
            )
        )
        chapter_headings = len(
            re.findall(
                r"(?im)^.{0,30}?Chương\s+[IVXLCDM\d]+",
                text,
            )
        )

        return {
            "text_length": len(text),
            "suspicious_total": suspicious_total,
            "suspicious_ratio": suspicious_total / text_length,
            "article_headings": article_headings,
            "chapter_headings": chapter_headings,
            **suspicious_characters,
        }

    @staticmethod
    def _fix_joined_words(text: str) -> str:
        """Tách các cặp từ pháp lý thường bị OCR dính liền.

        Chỉ thêm khoảng trắng tại các ranh giới có độ tin cậy cao. Danh
        sách được áp dụng nhiều lượt để sửa được chuỗi như
        ``chủsởhữucôngty`` thành ``chủ sở hữu công ty``.
        """
        word_boundaries = (
            ("chủ", "sở"),
            ("sở", "hữu"),
            ("hữu", "công"),
            ("cổ", "đông"),
            ("cổ", "phần"),
            ("hồ", "sơ"),
            ("điều", "lệ"),
            ("lệ", "là"),
            ("nghĩa", "vụ"),
            ("đề", "nghị"),
            ("nghị", "đăng"),
            ("nghị", "quyết"),
            ("trụ", "sở"),
            ("thị", "trường"),
            ("trừ", "trường"),
            ("tỷ", "lệ"),
            ("số", "lượng"),
            ("thủ", "tục"),
            ("trình", "tự"),
            ("thanh", "lý"),
            ("thành", "viên"),
            ("pháp", "luật"),
            ("đăng", "ký"),
            ("doanh", "nghiệp"),
            ("trách", "nhiệm"),
            ("thực", "hiện"),
            ("cụ", "thể"),
            ("chữ", "ký"),
            ("địa", "chỉ"),
            ("cơ", "sở"),
            ("giấy", "tờ"),
            ("tài", "sản"),
            ("vốn", "góp"),
            ("góp", "của"),
            ("phần", "của"),
            ("biểu", "quyết"),
            ("ủy", "quyền"),
            ("phổ", "thông"),
            ("chính", "xác"),
            ("liên", "lạc"),
            ("sau", "đây"),
            ("trở", "lên"),
            ("kể", "từ"),
            ("từ", "khi"),
            ("theo", "quy"),
            ("quy", "định"),
            ("công", "ty"),
            ("nhà", "nước"),
            ("bao", "gồm"),
            ("quyền", "và"),
            ("và", "nghĩa"),
            ("vụ", "của"),
            ("của", "chủ"),
            ("cho", "mỗi"),
            ("có", "thể"),
            ("được", "quyền"),
        )

        for _ in range(2):
            for left, right in word_boundaries:
                text = re.sub(
                    rf"(?i)(?<!\w)({left})({right})(?!\w)",
                    r"\1 \2",
                    text,
                )

                # Cũng tách ranh giới nằm trong chuỗi dài hơn; hai phía
                # phải là chữ để không tác động tới số hoặc dấu câu.
                text = re.sub(
                    rf"(?i)({left})(?={right})",
                    r"\1 ",
                    text,
                )

        return text

    @staticmethod
    def _fix_missing_diacritics(text: str) -> str:
        """Sửa cụm mất dấu khi ngữ cảnh pháp lý xác định được từ đúng."""
        phrase_corrections = {
            r"\bco\s+đông\b": "cổ đông",
            r"\bcố\s+đông\b": "cổ đông",
            r"\bco\s+phần\b": "cổ phần",
            r"\bcố\s+phần\b": "cổ phần",
            r"\bphan\s+vốn\b": "phần vốn",
            r"\bphan\s+góp\b": "phần góp",
            r"\bdia\s+chỉ\b": "địa chỉ",
            r"\btru\s+sở\b": "trụ sở",
            r"\bdoi\s+với\b": "đối với",
            r"\btrường\s+hop\b": "trường hợp",
            r"\bhữu\s+han\b": "hữu hạn",
            r"\bdai\s+diện\b": "đại diện",
            r"\bcó\s+thé\b": "có thể",
            r"\bđăng\s+ky\b": "đăng ký",
            r"\bloai\s+cổ\s+phần\b": "loại cổ phần",
            r"\bthe\s+quy\s+định\b": "theo quy định",
            r"\bdé\s+(?=thực hiện|bảo đảm|thành lập|tạo\b)": "để ",
        }

        for pattern, replacement in phrase_corrections.items():
            def replace_preserving_case(
                match: re.Match[str],
                corrected: str = replacement,
            ) -> str:
                if match.group(0)[:1].isupper():
                    return corrected[:1].upper() + corrected[1:]

                return corrected

            text = re.sub(
                pattern,
                replace_preserving_case,
                text,
                flags=re.IGNORECASE,
            )

        return text

    def _resolve_tessdata_path(
        self,
        tessdata_path: str | Path | None,
    ) -> Path:
        candidates: list[Path] = []

        # Ưu tiên đường dẫn được truyền trực tiếp.
        if tessdata_path is not None:
            candidates.append(Path(tessdata_path))

        # Sau đó dùng biến môi trường.
        env_path = os.getenv("TESSDATA_PREFIX")

        if env_path:
            env_candidate = Path(env_path)

            candidates.extend(
                [
                    env_candidate,
                    env_candidate / "tessdata",
                ]
            )

        # Thử tìm tesseract.exe từ PATH.
        executable = shutil.which("tesseract")

        if executable:
            executable_path = Path(executable).resolve()

            candidates.extend(
                [
                    executable_path.parent / "tessdata",
                    executable_path.parent.parent
                    / "share"
                    / "tessdata",
                    executable_path.parent.parent
                    / "share"
                    / "tesseract-ocr"
                    / "5"
                    / "tessdata",
                ]
            )

        # Chỉ là fallback cuối.
        candidates.extend(
            [
                Path(
                    r"C:\Program Files"
                    r"\Tesseract-OCR\tessdata"
                ),
                Path(
                    r"C:\Program Files (x86)"
                    r"\Tesseract-OCR\tessdata"
                ),
                Path(
                    "/usr/share/tesseract-ocr/5/tessdata"
                ),
                Path(
                    "/usr/share/tesseract-ocr/4.00/tessdata"
                ),
                Path("/usr/share/tessdata"),
                Path("/opt/homebrew/share/tessdata"),
                Path("/usr/local/share/tessdata"),
            ]
        )

        checked: list[Path] = []

        for candidate in candidates:
            candidate = candidate.expanduser()

            if candidate in checked:
                continue

            checked.append(candidate)

            if (
                candidate.exists()
                and candidate.is_dir()
                and any(candidate.glob("*.traineddata"))
            ):
                return candidate.resolve()

        searched = "\n".join(
            f"- {candidate}"
            for candidate in checked
        )

        raise FileNotFoundError(
            "Không tìm thấy thư mục tessdata.\n"
            f"Đã kiểm tra:\n{searched}\n\n"
            "Hãy thiết lập TESSDATA_PREFIX hoặc truyền "
            "tessdata_path vào PDFLoader."
        )

    def _validate_languages(self) -> None:
        languages = self.language.split("+")
        missing: list[Path] = []

        for language in languages:
            model_path = (
                self.tessdata_path
                / f"{language}.traineddata"
            )

            if not model_path.exists():
                missing.append(model_path)

        if missing:
            missing_text = "\n".join(
                f"- {path}"
                for path in missing
            )

            raise FileNotFoundError(
                "Thiếu model OCR:\n"
                f"{missing_text}"
            )
