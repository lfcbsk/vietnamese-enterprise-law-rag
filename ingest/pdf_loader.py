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
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.language = language
        self.ocr_dpi = ocr_dpi
        self.min_text_length = min_text_length
        self.use_cache = use_cache

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
        if self.use_cache and self.cache_path.exists():
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

                if len(native_text) >= self.min_text_length:
                    text = native_text

                    print(
                        f"Page {page_number}: "
                        f"native text ({len(text)} chars)"
                    )
                else:
                    print(
                        f"Page {page_number}: "
                        "đang OCR..."
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
    def clean_text(text: str) -> str:
        text = text.replace("\ufeff", "")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")

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

        # Giữ xuống dòng vì parser dùng cấu trúc dòng.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

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
