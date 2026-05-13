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

Paths for ``papers`` corpus and metadata roots come from ``.env`` (see `.env.example`).

Loads ``.env`` from this directory for **paths** plus **NCBI credentials**: ``NCBI_EMAIL``
(contact address for Entrez — often your regular email), optional ``NCBI_API_KEY``,
or aliases ``ENTREZ_EMAIL`` / ``ENTREZ_API_KEY``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

import papers_rag_config  # noqa: F401  — ensure `.env` loaded early

from abstract_extraction import (
    abstract_json_path,
    attach_pubmed_enrichment,
    extract_record_for_pdf,
    save_abstract_record,
)
from ncbi_pubmed import (
    fetch_pubmed_enrichment_for_doi,
    fetch_pubmed_enrichment_for_title,
)
from papers_paths import PAPERS_DIR, get_all_pdfs
from papers_rag_config import ABSTRACT_META_ROOT

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


ProgressCb = Callable[[float, str], None] | None


def should_skip_abstract_json(
    pdf_path: str,
    papers_dir: str,
    *,
    force: bool,
    only_missing: bool,
    refresh_if_newer_pdf: bool,
    abstract_meta_root: Path | None = None,
) -> bool:
    """When True, skip entire PDF (local extract + optional PubMed)."""
    if force:
        return False
    jpath = abstract_json_path(
        pdf_path, papers_dir, abstract_meta_root=abstract_meta_root
    )
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


def _apply_pubmed(
    rec: dict,
    *,
    ncbi_email: str | None,
    ncbi_key: str | None,
) -> None:
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


def pipeline_one_pdf(
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None,
    pubmed_meta: bool,
    force: bool,
    only_missing: bool,
    refresh_if_newer_pdf: bool,
    ncbi_email: str | None,
    ncbi_key: str | None,
) -> str:
    """
    Skip, extract+savelocal, optionally PubMed enrich. Returns ``"skipped"|"extracted"|"error"``.
    PubMed requires ``ncbi_email`` when ``pubmed_meta``.
    """
    try:
        if should_skip_abstract_json(
            pdf_path,
            papers_dir,
            force=force,
            only_missing=only_missing,
            refresh_if_newer_pdf=refresh_if_newer_pdf,
            abstract_meta_root=abstract_meta_root,
        ):
            return "skipped"

        rec = extract_record_for_pdf(
            pdf_path,
            papers_dir,
            abstract_meta_root=abstract_meta_root,
        )
        if pubmed_meta:
            if not ncbi_email:
                raise RuntimeError(
                    "PubMed requested but NCBI_EMAIL / ENTREZ_EMAIL is not set in .env"
                )
            _apply_pubmed(rec, ncbi_email=ncbi_email, ncbi_key=ncbi_key)
        save_abstract_record(
            rec,
            pdf_path,
            papers_dir,
            abstract_meta_root=abstract_meta_root,
        )
        return "extracted"
    except Exception as exc:
        print(f"\n[ERROR] {pdf_path}: {exc}", file=sys.stderr)
        return "error"


def run_abstract_extractions(
    *,
    papers_dir: str | None = None,
    abstract_meta_root: Path | None = None,
    pubmed_meta: bool = False,
    force: bool = False,
    only_missing: bool = False,
    refresh_if_newer_pdf: bool = False,
    limit: int = 0,
    progress_callback: ProgressCb = None,
) -> dict[str, int]:
    """
    Walk ``papers_dir`` with the same semantics as CLI ``main()``.

    Returns counts: extracted, skipped, errors.
    """
    root_pdf = papers_dir if papers_dir is not None else PAPERS_DIR
    meta_root = abstract_meta_root if abstract_meta_root is not None else ABSTRACT_META_ROOT

    pdfs = get_all_pdfs(root_pdf)
    if limit > 0:
        pdfs = pdfs[:limit]

    total = len(pdfs)
    ncbi_email = (
        os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL") or ""
    ).strip() or None
    ncbi_key = (
        os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY") or ""
    ).strip() or None

    if pubmed_meta and not ncbi_email:
        raise RuntimeError(
            "PubMed enabled but no NCBI contact email — set NCBI_EMAIL "
            "(or ENTREZ_EMAIL) in .env"
        )

    extracted = skipped = errors = 0

    def _pct_done(i_done: int) -> float:
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, i_done / float(total)))

    for i, pdf_path in enumerate(pdfs):
        if progress_callback:
            progress_callback(
                _pct_done(i),
                f"Processing {i + 1}/{total}: {Path(pdf_path).name}",
            )

        outcome = pipeline_one_pdf(
            pdf_path,
            root_pdf,
            abstract_meta_root=meta_root,
            pubmed_meta=pubmed_meta,
            force=force,
            only_missing=only_missing,
            refresh_if_newer_pdf=refresh_if_newer_pdf,
            ncbi_email=ncbi_email,
            ncbi_key=ncbi_key,
        )
        if outcome == "extracted":
            extracted += 1
        elif outcome == "skipped":
            skipped += 1
        else:
            errors += 1

    if progress_callback:
        progress_callback(1.0, "Done.")

    return {"extracted": extracted, "skipped": skipped, "errors": errors}


def main() -> int:
    """
    CLI entry: parse arguments, iterate PDFs, optionally attach PubMed enrichment,
    and write ``abstract_meta`` JSON. See module docstring for flags and ``.env`` keys.
    """
    ap = argparse.ArgumentParser(description="Extract abstracts → abstract_meta/*.json")
    ap.add_argument(
        "--papers-dir",
        default=None,
        help="Root folder of PDFs (default: PAPERS_DIR from .env)",
    )
    ap.add_argument(
        "--abstract-meta-root",
        default=None,
        help="Override ABSTRACT_META_ROOT from .env (absolute path)",
    )
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

    papers_dir = args.papers_dir if args.papers_dir is not None else PAPERS_DIR
    meta_root: Path | None = None
    if args.abstract_meta_root:
        meta_root = Path(os.path.expanduser(args.abstract_meta_root)).resolve()

    pdfs = get_all_pdfs(papers_dir)
    if args.limit:
        pdfs = pdfs[: args.limit]

    ncbi_email = (
        os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL") or ""
    ).strip() or None
    ncbi_key = (
        os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY") or ""
    ).strip() or None

    print(f"Papers dir: {papers_dir}")
    print(f"PDFs to process: {len(pdfs)}")
    if args.abstract_meta_root:
        print(f"abstract_meta_root (override): {meta_root}")
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

    if args.pubmed_meta and not ncbi_email:
        print("\n[ERROR] PubMed requested but NCBI_EMAIL / ENTREZ_EMAIL not set.", file=sys.stderr)
        return 2

    extracted = skipped = errors = 0

    try:
        for pdf_path in tqdm(pdfs, desc="Abstracts"):
            outcome = pipeline_one_pdf(
                pdf_path,
                papers_dir,
                abstract_meta_root=meta_root,
                pubmed_meta=args.pubmed_meta,
                force=args.force,
                only_missing=args.only_missing,
                refresh_if_newer_pdf=args.refresh_if_newer_pdf,
                ncbi_email=ncbi_email,
                ncbi_key=ncbi_key,
            )
            if outcome == "extracted":
                extracted += 1
            elif outcome == "skipped":
                skipped += 1
            else:
                errors += 1
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    print(
        "Done.",
        f"extracted={extracted} skipped={skipped} errors={errors}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
