import json
from pathlib import Path

from pdf_loader import PDFLoader
from parser import VietnameseLawParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "law_chunks.json"


def run_ingestion() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Không tìm thấy file PDF trong {DATA_DIR}"
        )

    parser = VietnameseLawParser()
    all_chunks = []

    for pdf_file in pdf_files:
        source = pdf_file.name

        print(f"Loading: {source}")

        loader = PDFLoader(str(pdf_file))
        raw_text = loader.load_and_clean()

        chunks = parser.parse(
            raw_text,
            source=source,
        )

        all_chunks.extend(chunks)

        print(f"Extracted: {len(chunks)} chunks")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_ingestion()