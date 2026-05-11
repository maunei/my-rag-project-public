#!/usr/bin/env python3
"""
CLI: extract abstract metadata JSON for every PDF under PAPERS_DIR.

Usage:
    conda activate papers_rag
    cd .../papers-rag_app
    python extract_abstracts.py                    # all PDFs
    python extract_abstracts.py --limit 10         # smoke test
    python extract_abstracts.py --only-missing   # skip if JSON already exists
    python extract_abstracts.py --only-missing --refresh-if-newer-pdf
    python extract_abstracts.py --pubmed-meta    # NCBI: `{doi}[doi]` from PDF metadata/links first; title `[Title]` fallback

Writes mirrored JSON under ``abstract_meta/`` (`abstract_text` from PDF, `doi_*` from PDF metadata,
links, early text; `abstract_pubmed` from PubMed when using ``--pubmed-meta``, DOI-first when available).
Does not modify ChromaDB.
Loads ``.env`` from this directory for **NCBI credentials**: ``NCBI_EMAIL``
(contact address for Entrez — often your normal email), optional ``NCBI_API_KEY``,
or aliases ``ENTREZ_EMAIL`` / ``ENTREZ_API_KEY``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from abstract_extraction import (  # noqa: E402  (after load_dotenv)
    abstract_json_path,
    attach_pubmed_enrichment,
    extract_record_for_pdf,
    save_abstract_record,
)
from ncbi_pubmed import (  # noqa: E402
    fetch_pubmed_enrichment_for_doi,
    fetch_pubmed_enrichment_for_title,
)
from papers_paths import PAPERS_DIR, get_all_pdfs  # noqa: E402

_DOI_ROUTE_FAILURE = frozenset(
    {
        "no_hit",
        "http_error",
        "parse_error",
        "ncbi_esearch_error",
        "skipped_missing_doi",
        "skipped_invalid_doi",
    }
)


def should_skip_abstract_json(
    pdf_path: str,
    papers_dir: str,
    *,
    force: bool,
    only_missing: bool,
    refresh_if_newer_pdf: bool,
) -> bool:
    """When True, skip entire PDF (local extract + optional PubMed)."""
    if force:
        return False
    jpath = abstract_json_path(pdf_path, papers_dir)
    if not jpath.is_file():
        return False
    if refresh_if_newer_pdf:
        try:
            if Path(pdf_path).stat().st_mtime > jpath.stat().st_mtime:
                return False
        except OSError:
            return False
    if only_missing:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract abstracts → abstract_meta/*.json")
    ap.add_argument("--papers-dir", default=PAPERS_DIR, help="Root folder of PDFs")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N PDFs (0=all)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract every PDF (ignore only-missing / refresh rules)",
    )
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip PDFs that already have an abstract JSON (no local or PubMed updates)",
    )
    ap.add_argument(
        "--refresh-if-newer-pdf",
        action="store_true",
        help="With --only-missing: still process when the PDF is newer than its JSON",
    )
    ap.add_argument(
        "--pubmed-meta",
        action="store_true",
        help="After extraction, query NCBI E-utilities (title from PDF or title_guess)",
    )
    args = ap.parse_args()

    pdfs = get_all_pdfs(args.papers_dir)
    if args.limit:
        pdfs = pdfs[: args.limit]

    ncbi_email = (os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL") or "").strip() or None
    ncbi_key = (os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY") or "").strip() or None

    print(f"Papers dir: {args.papers_dir}")
    print(f"PDFs to process: {len(pdfs)}")
    if args.only_missing:
        print("Mode: --only-missing", end="")
        if args.refresh_if_newer_pdf:
            print(" + --refresh-if-newer-pdf")
        else:
            print()
    if args.pubmed_meta:
        print(
            f"NCBI credentials (Entrez): contact_email="
            f"{'set (NCBI_EMAIL or ENTREZ_EMAIL)' if ncbi_email else 'missing'}, "
            f"api_key={'set' if ncbi_key else 'optional/missing'}"
        )

    for pdf_path in tqdm(pdfs, desc="Abstracts"):
        try:
            if should_skip_abstract_json(
                pdf_path,
                args.papers_dir,
                force=args.force,
                only_missing=args.only_missing,
                refresh_if_newer_pdf=args.refresh_if_newer_pdf,
            ):
                continue

            rec = extract_record_for_pdf(pdf_path, args.papers_dir)
            if args.pubmed_meta:
                qdoi = (rec.get("doi_for_pubmed") or "").strip()
                enc = {}
                ncbi_abstract = ""
                efetch_err = None
                doi_failed = False
                if qdoi:
                    enc, ncbi_abstract, efetch_err = fetch_pubmed_enrichment_for_doi(
                        qdoi, email=ncbi_email, api_key=ncbi_key
                    )
                    doi_failed = enc.get("status") in _DOI_ROUTE_FAILURE
                if not qdoi or doi_failed:
                    qtitle = (rec.get("title_pdf") or rec.get("title_guess") or "").strip()
                    enc, ncbi_abstract, efetch_err = fetch_pubmed_enrichment_for_title(
                        qtitle, email=ncbi_email, api_key=ncbi_key
                    )

                attach_pubmed_enrichment(rec, enc)
                rec["abstract_pubmed"] = (ncbi_abstract or "").strip()
                rec.pop("pubmed_efetch_error", None)
                if efetch_err:
                    rec["pubmed_efetch_error"] = efetch_err
            save_abstract_record(rec, pdf_path, args.papers_dir)
        except Exception as exc:
            print(f"\n[ERROR] {pdf_path}: {exc}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
