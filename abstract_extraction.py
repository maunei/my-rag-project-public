"""
abstract_extraction.py — Local PDF abstract extraction → mirrored JSON files.

Writes one JSON per PDF under abstract_meta/<rel_path>/<stem>_abstract.json
mirroring PAPERS_DIR. No Vertex / API calls.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz

SCHEMA_VERSION = 1
EXTRACTION_VERSION = "1"
LANGUAGE_DETECTED = "en"
PLACEHOLDER_NOT_DETECTED = "Abstract not detected."

APP_DIR = Path(__file__).resolve().parent
ABSTRACT_META_ROOT = APP_DIR / "abstract_meta"
EXPORTED_PROMPTS_DIR = APP_DIR / "exported_prompts"

# ── Paths ─────────────────────────────────────────────────────────────────────


def abstract_json_path(pdf_path: str, papers_dir: str) -> Path:
    rel = Path(pdf_path).relative_to(Path(papers_dir))
    return ABSTRACT_META_ROOT / rel.parent / f"{rel.stem}_abstract.json"


def ensure_export_dir() -> Path:
    EXPORTED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORTED_PROMPTS_DIR


# ── Title helper (match indexer) ─────────────────────────────────────────────

def readable_title(filepath: str) -> str:
    stem = Path(filepath).stem
    parts = stem.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        stem = parts[1]
    return stem.replace("_", " ").title()


# ── Extraction logic ──────────────────────────────────────────────────────────

_STOP_HEADERS = re.compile(
    r"(?im)^\s*(introduction|keywords|background|significance|highlights|"
    r"abbreviations|conflict\s+of\s+interest|references)\s*$"
)

_COPYRIGHTISH = re.compile(
    r"(?i)\b(copyright\s+©|all rights reserved|doi:|http://dx\.doi\.org|arxiv:)\b"
)


def _early_text(pdf_path: str, max_pages: int = 4) -> tuple[str, list[int]]:
    warnings: list[str] = []
    pages_used: list[int] = []
    chunks: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        n = min(max_pages, len(doc))
        for i in range(n):
            pages_used.append(i + 1)
            chunks.append(doc.load_page(i).get_text("text"))
        doc.close()
    except Exception as exc:
        warnings.append(f"pdf_read_error:{exc}")
        return "", pages_used
    blob = "\n\n".join(chunks)
    if len(blob.strip()) < 80:
        warnings.append("very_little_text_early_pages")
    return blob, pages_used


def _extract_after_heading(blob: str) -> tuple[str | None, str | None]:
    """Try explicit English Abstract / Summary section. Returns (text, source_tag)."""
    # Match a line that is mostly just "Abstract" or "Summary"
    header = re.compile(r"(?im)^\s*(abstract|summary)\s*$")
    m = header.search(blob)
    if not m:
        return None, None
    start = m.end()
    rest = blob[start:].strip()
    # Cut at next major section
    stop_m = _STOP_HEADERS.search(rest)
    if stop_m:
        rest = rest[: stop_m.start()].strip()
    # Trim junk at end (running into copyright lines)
    lines = rest.splitlines()
    kept: list[str] = []
    for line in lines:
        if _COPYRIGHTISH.search(line):
            break
        kept.append(line)
    text = "\n".join(kept).strip()
    if len(text) < 40:
        return None, None
    tag = f"explicit_heading:{m.group(1).lower()}"
    return text, tag


def _extract_heuristic(blob: str) -> tuple[str | None, str | None]:
    """First substantial paragraph before Introduction — low confidence."""
    # Drop header-ish noise: take slice after first blank double newline chunk sometimes helps
    intro_m = re.compile(r"(?im)^\s*introduction\s*$").search(blob)
    window = blob[: intro_m.start()] if intro_m else blob[:6000]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", window) if len(p.strip()) > 120]
    for para in paragraphs:
        low = para.lower()
        if low.startswith(("abstract", "summary")):
            continue
        if _COPYRIGHTISH.search(para):
            continue
        # Skip author affiliation blocks (many short lines / emails)
        if "@" in para and para.count("@") >= 2:
            continue
        return para, "first_pages_heuristic"
    return None, None


def extract_record_for_pdf(pdf_path: str, papers_dir: str) -> dict:
    """Build full JSON-serializable record for one PDF."""
    warnings: list[str] = []
    pdf_p = Path(pdf_path)
    try:
        rel_path = str(pdf_p.relative_to(Path(papers_dir)))
    except ValueError:
        rel_path = pdf_p.name

    try:
        mtime = datetime.fromtimestamp(pdf_p.stat().st_mtime, tz=timezone.utc)
        pdf_mtime_iso = mtime.isoformat()
    except OSError:
        pdf_mtime_iso = ""

    indexed_at_iso = datetime.now(timezone.utc).isoformat()
    title_guess = readable_title(pdf_path)

    blob, pages_span_list = _early_text(pdf_path)
    pages_span = {"start_page": min(pages_span_list), "end_page": max(pages_span_list)} if pages_span_list else {}

    abstract_text = PLACEHOLDER_NOT_DETECTED
    status = "not_detected"
    source = "none"

    if blob:
        ext, src = _extract_after_heading(blob)
        if ext:
            abstract_text = ext
            status = "extracted"
            source = src or "explicit_heading"
        else:
            ext2, src2 = _extract_heuristic(blob)
            if ext2:
                abstract_text = ext2
                status = "low_confidence"
                source = src2 or "first_pages_heuristic"
                warnings.append("no_explicit_abstract_heading")

    word_count = len(abstract_text.split())
    char_count = len(abstract_text)

    return {
        "schema_version": SCHEMA_VERSION,
        "file_path": str(pdf_p.resolve()),
        "rel_path": rel_path,
        "file_name": pdf_p.name,
        "abstract_text": abstract_text,
        "status": status,
        "source": source,
        "pages_span": pages_span,
        "extraction_version": EXTRACTION_VERSION,
        "language_detected": LANGUAGE_DETECTED,
        "char_count": char_count,
        "word_count": word_count,
        "pdf_mtime_iso": pdf_mtime_iso,
        "indexed_at_iso": indexed_at_iso,
        "title_guess": title_guess,
        "warnings": warnings,
    }


def save_abstract_record(record: dict, pdf_path: str, papers_dir: str) -> Path:
    path = abstract_json_path(pdf_path, papers_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def load_abstract_record(pdf_path: str, papers_dir: str) -> dict | None:
    path = abstract_json_path(pdf_path, papers_dir)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_and_save_pdf(pdf_path: str, papers_dir: str) -> dict:
    rec = extract_record_for_pdf(pdf_path, papers_dir)
    save_abstract_record(rec, pdf_path, papers_dir)
    return rec
