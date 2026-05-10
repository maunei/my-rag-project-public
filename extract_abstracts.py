#!/usr/bin/env python3
"""
CLI: extract abstract metadata JSON for every PDF under PAPERS_DIR.

Usage:
    conda activate papers_rag
    cd .../papers-rag_app
    python extract_abstracts.py              # all PDFs
    python extract_abstracts.py --limit 10   # smoke test

Writes mirrored files under abstract_meta/ — does not modify ChromaDB.
"""

from __future__ import annotations

import argparse
import sys

from tqdm import tqdm

from papers_paths import PAPERS_DIR, get_all_pdfs
from abstract_extraction import extract_and_save_pdf


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract abstracts → abstract_meta/*.json")
    ap.add_argument("--papers-dir", default=PAPERS_DIR, help="Root folder of PDFs")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N PDFs (0=all)")
    args = ap.parse_args()

    pdfs = get_all_pdfs(args.papers_dir)
    if args.limit:
        pdfs = pdfs[: args.limit]

    print(f"Papers dir: {args.papers_dir}")
    print(f"PDFs to process: {len(pdfs)}")

    for pdf_path in tqdm(pdfs, desc="Abstracts"):
        try:
            extract_and_save_pdf(pdf_path, args.papers_dir)
        except Exception as exc:
            print(f"\n[ERROR] {pdf_path}: {exc}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
