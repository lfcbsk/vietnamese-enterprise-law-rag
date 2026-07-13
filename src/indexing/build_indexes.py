from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.indexing import build_bm25_index, build_dense_index
from src.indexing.embedder import EmbedderConfig, LawEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "indexes" / "manifest.json"


def run(*, skip_dense: bool = False, skip_bm25: bool = False) -> dict:
    manifest: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    if not skip_dense:
        print("\n" + "=" * 80)
        print("Building dense (ChromaDB) index...")

        embedder = LawEmbedder(EmbedderConfig())
        manifest["dense_index"] = build_dense_index.build(embedder=embedder)

        print(json.dumps(manifest["dense_index"], ensure_ascii=False, indent=2))

    if not skip_bm25:
        print("\n" + "=" * 80)
        print("Building BM25 index...")

        manifest["bm25_index"] = build_bm25_index.build()

        print(json.dumps(manifest["bm25_index"], ensure_ascii=False, indent=2))

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"\nManifest saved to: {MANIFEST_PATH}")

    return manifest


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    argument_parser = argparse.ArgumentParser(
        description="Build dense (ChromaDB) và BM25 index từ data/law_chunks.json",
    )
    argument_parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Bỏ qua bước build dense index (chỉ build lại BM25).",
    )
    argument_parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Bỏ qua bước build BM25 index (chỉ build lại dense).",
    )
    arguments = argument_parser.parse_args()

    run(skip_dense=arguments.skip_dense, skip_bm25=arguments.skip_bm25)