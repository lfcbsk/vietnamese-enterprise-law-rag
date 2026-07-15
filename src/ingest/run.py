from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Support both `python -m src.ingest.run` and `python src/ingest/run.py`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.parser import VietnameseLawParser  # noqa: E402
from src.ingest.pdf_loader import PDFLoader  # noqa: E402
from src.ingest.segment_fix import (  # noqa: E402
    build_syllable_dict,
    fix_text,
)


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "law_chunks.json"


def validate_chunks(
    chunks: list[dict],
) -> None:
    if not chunks:
        raise ValueError(
            "Không tạo được chunk nào. "
            "Hãy kiểm tra OCR output và regex parser."
        )

    chunk_ids = [
        chunk["id"]
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count
        in Counter(chunk_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            f"Có ID trùng: {duplicate_ids[:10]}"
        )

    empty_chunks = [
        chunk["id"]
        for chunk in chunks
        if not chunk["content"].strip()
    ]

    if empty_chunks:
        raise ValueError(
            f"Có chunk rỗng: {empty_chunks[:10]}"
        )


def run_ingestion(*, force_ocr: bool = False) -> None:
    # PowerShell trên Windows đôi khi dùng cp1252 và không in được
    # tiếng Việt, dù dữ liệu đầu ra vẫn được ghi đúng bằng UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Không tìm thấy PDF trong {DATA_DIR}"
        )

    parser = VietnameseLawParser(
        law_name="Luật Doanh nghiệp 2020"
    )

    loaded_documents: list[tuple[str, str]] = []

    for pdf_file in pdf_files:
        source = pdf_file.name

        print("\n" + "=" * 80)
        print(f"Loading: {source}")

        loader = PDFLoader(
            pdf_file,
            language="vie+eng",
            ocr_dpi=300,
            min_text_length=80,
            use_cache=True,
            # PDF có lớp text lỗi font (dºanh, hºặc, theº...), vì vậy
            # --force-ocr sẽ OCR ảnh toàn bộ trang và ghi đè cache cũ.
            force_ocr=force_ocr,
        )

        raw_text = loader.load_and_clean()

        print(
            f"Extracted text length: "
            f"{len(raw_text)} characters"
        )

        loaded_documents.append((source, raw_text))

    syllable_frequency = build_syllable_dict(
        [text for _, text in loaded_documents]
    )
    all_chunks: list[dict] = []

    for source, raw_text in loaded_documents:
        fixed_text, segment_changes = fix_text(
            raw_text,
            syllable_frequency,
        )

        print(
            f"Segment fix ({source}): "
            f"{len(segment_changes)} token(s)"
        )
        for before, after in segment_changes[:10]:
            print(f"  {before!r} -> {after!r}")

        chunks = parser.parse(fixed_text, source=source)

        print(
            f"Extracted: {len(chunks)} chunks"
        )

        if not chunks:
            print("\nOCR preview:")
            print(raw_text[:2000])

        all_chunks.extend(chunks)

    validate_chunks(all_chunks)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst 3 chunks:")

    for chunk in all_chunks[:3]:
        print("-" * 80)
        print(chunk["id"])
        print(
            chunk["chapter"],
            chunk["article"],
            chunk["article_title"],
        )
        print(chunk["content"][:500])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Bỏ qua cache/native text và OCR lại toàn bộ PDF.",
    )
    arguments = argument_parser.parse_args()

    run_ingestion(force_ocr=arguments.force_ocr)
