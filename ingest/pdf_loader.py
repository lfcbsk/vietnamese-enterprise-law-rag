from __future__ import annotations

import os
import re
from pathlib import Path

import pymupdf


class PDFLoader:
    """
    Đọc PDF có text layer và tự động OCR các trang scan bằng Tesseract.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        *,
        language: str = "vie+eng",
        ocr_dpi: int = 300,
        min_text_length: int = 80,
        tessdata_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        force_ocr: bool = False,
        max_pages: int | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path).resolve()
        self.language = language
        self.ocr_dpi = ocr_dpi
        self.min_text_length = min_text_length
        self.use_cache = use_cache
        self.force_ocr = force_ocr
        self.max_pages = max_pages

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy PDF: {self.pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File không phải PDF: {self.pdf_path}")

        self.tessdata_path = self._resolve_tessdata_path(tessdata_path)

        if cache_dir is None:
            cache_dir = self.pdf_path.parent / "ocr"

        self.cache_dir = Path(cache_dir).resolve()
        self.cache_path = self.cache_dir / f"{self.pdf_path.stem}.txt"

        self._validate_language_files()

    def load_and_clean(self) -> str:
        if self.use_cache and self.cache_path.exists():
            print(f"Using OCR cache: {self.cache_path}")
            return self.cache_path.read_text(encoding="utf-8")

        document = pymupdf.open(self.pdf_path)
        page_texts: list[str] = []

        try:
            total_pages = len(document)
            if self.max_pages is not None:
                total_pages = min(total_pages, self.max_pages)

            for page_index in range(total_pages):
                page = document[page_index]
                page_number = page_index + 1

                text = self._extract_page_text(
                    page=page,
                    page_number=page_number,
                )

                cleaned_text = self.clean_text(text)
                page_texts.append(
                    f"[PAGE {page_number}]\n{cleaned_text}"
                )

        finally:
            document.close()

        full_text = "\n\n".join(page_texts).strip()

        if not full_text:
            raise RuntimeError(
                "Không lấy được text nào từ PDF. "
                "Hãy kiểm tra Tesseract, tessdata và chất lượng file scan."
            )

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(full_text, encoding="utf-8")
            print(f"Saved OCR cache: {self.cache_path}")

        return full_text

    def _extract_page_text(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
    ) -> str:
        native_text = page.get_text("text", sort=True).strip()

        should_ocr = (
            self.force_ocr
            or len(native_text) < self.min_text_length
        )

        if not should_ocr:
            print(
                f"Page {page_number}: "
                f"native text ({len(native_text)} chars)"
            )
            return native_text

        print(
            f"Page {page_number}: "
            f"OCR ({len(native_text)} native chars)"
        )

        try:
            text_page = page.get_textpage_ocr(
                language=self.language,
                dpi=self.ocr_dpi,
                full=True,
                tessdata=str(self.tessdata_path),
            )

            ocr_text = page.get_text(
                "text",
                textpage=text_page,
                sort=True,
            ).strip()

        except Exception as exc:
            raise RuntimeError(
                f"OCR thất bại ở trang {page_number}. "
                f"Tessdata: {self.tessdata_path}. "
                f"Lỗi gốc: {exc}"
            ) from exc

        print(
            f"Page {page_number}: "
            f"OCR output ({len(ocr_text)} chars)"
        )
        return ocr_text

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\ufeff", "")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")

        # Giữ xuống dòng để parser nhận diện Chương / Điều / Khoản.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    import os
import shutil
from pathlib import Path


def _resolve_tessdata_path(
    self,
    tessdata_path: str | Path | None,
) -> Path:
    candidates: list[Path] = []

    # 1. Đường dẫn do người dùng truyền vào
    if tessdata_path is not None:
        candidates.append(Path(tessdata_path))

    # 2. Biến môi trường
    env_path = os.getenv("TESSDATA_PREFIX")

    if env_path:
        env_candidate = Path(env_path)

        # Hỗ trợ cả hai cách:
        # TESSDATA_PREFIX=/path/to/tessdata
        # TESSDATA_PREFIX=/path/to/tesseract
        candidates.extend(
            [
                env_candidate,
                env_candidate / "tessdata",
            ]
        )

    # 3. Tự tìm executable từ PATH
    tesseract_executable = shutil.which("tesseract")

    if tesseract_executable:
        executable_path = Path(tesseract_executable).resolve()

        candidates.extend(
            [
                executable_path.parent / "tessdata",
                executable_path.parent.parent / "share" / "tessdata",
                executable_path.parent.parent
                / "share"
                / "tesseract-ocr"
                / "5"
                / "tessdata",
            ]
        )

    # 4. Fallback phổ biến theo hệ điều hành
    candidates.extend(
        [
            # Windows
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),

            # Linux phổ biến
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/share/tessdata"),

            # macOS Homebrew
            Path("/opt/homebrew/share/tessdata"),
            Path("/usr/local/share/tessdata"),
        ]
    )

    # Loại bỏ đường dẫn trùng
    unique_candidates: list[Path] = []

    for candidate in candidates:
        candidate = candidate.expanduser()

        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    # Kiểm tra thư mục có thực sự chứa traineddata
    for candidate in unique_candidates:
        if (
            candidate.exists()
            and candidate.is_dir()
            and any(candidate.glob("*.traineddata"))
        ):
            return candidate.resolve()

    searched = "\n".join(
        f"- {candidate}"
        for candidate in unique_candidates
    )

    raise FileNotFoundError(
        "Không tìm thấy thư mục tessdata.\n\n"
        "Các đường dẫn đã kiểm tra:\n"
        f"{searched}\n\n"
        "Hãy truyền tessdata_path trực tiếp hoặc thiết lập "
        "biến môi trường TESSDATA_PREFIX."
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = project_root / "data" / "59.signed.pdf"

    loader = PDFLoader(
        pdf_path,
        language="vie+eng",
        ocr_dpi=300,
        min_text_length=80,
        use_cache=True,
        force_ocr=False,
        max_pages=3,  # Đổi thành None để chạy toàn bộ PDF.
    )

    extracted_text = loader.load_and_clean()

    print("\n" + "=" * 80)
    print("PREVIEW")
    print("=" * 80)
    print(extracted_text[:3000])