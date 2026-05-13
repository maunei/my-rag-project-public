"""
abstract_extraction.py — Local PDF abstract extraction → mirrored JSON files.

Writes one JSON per PDF under abstract_meta/<rel_path>/<basename>.json
mirroring PAPERS_DIR. No Vertex / API calls.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz

SCHEMA_VERSION = 1
EXTRACTION_VERSION = "4"
LANGUAGE_DETECTED = "en"
PLACEHOLDER_NOT_DETECTED = "Abstract not detected."

# DOI normalization / extraction (publisher metadata + link annotations + early text).
_DOI_IN_TEXT = re.compile(
    r"(?i)\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s\])>;,\n]+)"
)

from papers_rag_config import ABSTRACT_META_ROOT, EXPORTED_PROMPTS_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────


def abstract_json_path(
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None = None,
) -> Path:
    """Mirrored sidescar JSON path under configured ``abstract_meta`` root."""
    root = abstract_meta_root if abstract_meta_root is not None else ABSTRACT_META_ROOT
    rel = Path(pdf_path).relative_to(Path(papers_dir))
    return root / rel.parent / f"{rel.stem}.json"


def ensure_export_dir() -> Path:
    """Create ``exported_prompts/`` if missing (used for manual LLM export bundles)."""
    EXPORTED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORTED_PROMPTS_DIR


def _truncate_doi_tail(s: str) -> str:
    """Strip punctuation often glued to bare DOIs in PDF text streams."""
    t = s.strip()
    while t and t[-1] in ".,;:!?)]}\"'›»":
        t = t[:-1]
    while t.endswith("\\"):
        t = t[:-1]
    return t.strip()


def normalize_doi(s: str) -> str | None:
    """Return lowercase normalized DOI (no URL / doi: prefix), or None if invalid."""
    if not (s and str(s).strip()):
        return None
    t = str(s).strip()
    t = re.sub(r"^\s*doi:\s*", "", t, flags=re.I)
    t = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", t, flags=re.I)
    m = _DOI_IN_TEXT.search(t) if "10." in t else None
    if m:
        t = m.group(1)
    t = _truncate_doi_tail(t.split()[0])
    # Minimal validation: registrar prefix + non-empty suffix
    if not re.match(r"(?i)10\.\d{4,9}/\S+", t):
        return None
    return t.lower()


_SUPPLEMENTARY_DOI_PREFIXES = (
    "10.5281/zenodo",
    "10.6084/",
    "10.7937/",
    "10.25407/",
)


def _doi_pubmed_preference_rank(d: str) -> tuple[int, str]:
    """
    Lower first element = better default for PMID lookup (prefer journal bundles over data repos).

    Stable tiebreaker: lexical DOI order.
    """
    low = d.lower()
    if any(low.startswith(p) for p in _SUPPLEMENTARY_DOI_PREFIXES):
        return (300, low)
    if low.startswith("10.1101/"):  # preprint — often PubMed-listed
        return (80, low)
    if low.startswith(
        (
            "10.1038/",
            "10.1016/",
            "10.1073/",
            "10.1126/",
            "10.7554/",
            "10.1371/",
            "10.3389/",
            "10.1093/",
            "10.1242/",
            "10.1186/",
            "10.1158/",
            "10.1007/",
            "10.7717/",
            "10.3791/",
        )
    ):
        return (0, low)
    return (40, low)


def pdf_title_suggests_correction(title: str) -> bool:
    """Heuristic for Springer/Nature correction PDFs carrying two sibling DOIs."""
    t = (title or "").strip().lower()
    if not t:
        return False
    return (
        t.startswith(
            ("author correction:", "publisher correction:", "corrigendum:", "erratum:", "withdrawal:")
        )
        or "author correction:" in t
        or "publisher correction:" in t
    )


def _metadata_canonical_doi(md: dict) -> tuple[str | None, str]:
    """
    Prefer DOI spelled in Document Info ``subject`` line; else scan other metadata strings.
    Returns (doi_or_none, source_tag).
    """
    items: list[tuple[str, str]] = []
    for key in ("subject", "keywords", "title", "producer", "creator", "copyright"):
        v = (md.get(key) or "").strip()
        if v:
            items.append((key, v))

    prio = {"subject": 0, "keywords": 1, "title": 2}

    def sort_key(kv: tuple[str, str]):
        """Order metadata keys for DOI scanning (subject before keywords before title)."""
        return (prio.get(kv[0], 5), kv[0])

    for key, blob in sorted(items, key=sort_key):
        m = re.search(r"(?i)doi:\s*(10\.\d{4,9}/[^\s,)]+)", blob)
        if m:
            d = normalize_doi(m.group(1))
            if d:
                return d, f"metadata:{key}"
    for key, blob in sorted(items, key=sort_key):
        for cand in _DOI_IN_TEXT.findall(blob):
            d = normalize_doi(cand)
            if d:
                return d, f"metadata:{key}_bare"
    return None, ""


def _link_dois_from_doc(doc: fitz.Document, max_pages: int = 6) -> list[tuple[str, str]]:
    """Collect DOIs from ``doi.org`` hyperlink URIs on the first ``max_pages`` (deduplicated, with page tags)."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    n = min(len(doc), max_pages)
    for i in range(n):
        links = []
        try:
            links = doc.load_page(i).get_links() or []
        except Exception:
            continue
        for ln in links:
            uri = (ln.get("uri") or "").strip()
            if "doi.org" not in uri.lower():
                continue
            m = _DOI_IN_TEXT.search(uri) or re.search(
                r"(?i)doi\.org/(10\.\d{4,9}/[^\s?&\)]+)", uri
            )
            raw = ""
            if m:
                raw = m.group(1)
            elif "doi.org/" in uri.lower():
                after = uri.split("doi.org/", 1)[-1]
                raw = after.split("?", 1)[0]
            if not raw:
                continue
            d = normalize_doi(raw)
            if d and d not in seen:
                seen.add(d)
                out.append((d, f"link:p{i+1}"))
    return out


def extract_dois_for_pdf_path(pdf_path: str, title_pdf: str) -> dict:
    """
    Collect DOIs from PDF document info, hyperlink annotations (doi.org URIs),
    and the first pages of extracted text.

    Computes ``doi_for_pubmed`` using a simple rule: corrections / errata bundle two DOIs
    (publisher record + original article); for RAG we prefer the cited article DOI by dropping
    the metadata-canonical prism DOI whenever the title implies a correction and another
    journal DOI survives — otherwise we prefer canonical + journal-ranked fallbacks.

    Returned keys mirror fields written into abstract JSON records.
    """
    detail: list[dict[str, str]] = []
    ordered: list[str] = []
    seen: set[str] = set()

    def push(d_raw: str, source: str) -> None:
        """Normalize ``d_raw``, append to ``detail``, and extend ``ordered`` once per unique DOI."""
        d = normalize_doi(d_raw)
        if not d:
            return
        detail.append({"doi": d, "source": source})
        if d not in seen:
            seen.add(d)
            ordered.append(d)

    warnings: list[str] = []

    canonical: str | None = None
    canonical_src = ""

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        warnings.append(f"doi_open_error:{exc!s}")
        return {
            "doi_canonical_pdf": "",
            "doi_candidates": [],
            "doi_for_pubmed": "",
            "doi_selection_note": "open_failed",
            "doi_extraction_warnings": warnings,
        }

    try:
        md = doc.metadata or {}
        canonical, canonical_src = _metadata_canonical_doi(md)

        # Metadata values may carry DOIs missed by naive subject scan — collect all.
        for key in ("subject", "keywords", "title", "producer", "creator"):
            blob = md.get(key) or ""
            for m in _DOI_IN_TEXT.finditer(blob):
                push(m.group(1), f"metadata_xml:{key}")
            bare = re.findall(r"(?<![\w./])(10\.\d{4,9}/[^\s\])>;,\n]+)", str(blob))
            for frag in bare:
                push(frag, f"metadata_fallback:{key}")

        for doi, src in _link_dois_from_doc(doc):
            push(doi, src)

        n = min(len(doc), 4)
        for i in range(n):
            blob = ""
            try:
                blob = doc.load_page(i).get_text("text") or ""
            except Exception as exc:
                warnings.append(f"doi_page_text:{i+1}:{exc!s}")
                continue
            for m in _DOI_IN_TEXT.finditer(blob):
                push(m.group(1), f"text:p{i+1}")

        title_hint = pdf_title_suggests_correction(title_pdf)

        picked = ""
        note = ""

        if not ordered:
            picked = ""
            note = "no_doi_detected"

        elif title_hint and canonical and any(d != canonical for d in ordered):
            others = [d for d in ordered if d != canonical]
            others.sort(key=_doi_pubmed_preference_rank)
            picked = others[0]
            note = (
                "correction_pdf_prefer_noncanonical_doi"
                f";prism_canonical={canonical};picked={picked}"
            )

        elif canonical:
            picked = canonical
            note = f"canonical_metadata:{canonical_src or 'unknown'}"

        else:
            ranked = sorted(ordered, key=_doi_pubmed_preference_rank)
            picked = ranked[0]
            note = "no_canonical_journal_rank_fallback"

        canonical_store = canonical or ""

    finally:
        doc.close()

    return {
        "doi_canonical_pdf": canonical_store,
        "doi_candidates": detail,
        "doi_for_pubmed": picked,
        "doi_selection_note": note,
        "doi_extraction_warnings": warnings,
    }


# ── Title helper (match indexer) ─────────────────────────────────────────────

def readable_title(filepath: str) -> str:
    """Human-ish title from path stem (strip leading numeric prefix from indexer naming; title-case words)."""
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
    """Concatenate raw text from the first ``max_pages`` for heading / heuristic abstract extraction."""
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


def extract_document_title(pdf_path: str) -> tuple[str, str, list[str]]:
    """
    Best-effort title from PDF metadata or first-page text before 'Abstract'.

    Returns (title_pdf, title_source, extra_warnings).
    ``title_source`` is one of ``metadata``, ``pdf_heuristic``, ``none``.
    """
    warnings: list[str] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return "", "none", [f"title_open_error:{exc!s}"]
    try:
        md = doc.metadata or {}
        meta_t = (md.get("title") or "").strip()
        bad_meta = {"", "untitled", "untitled document", "no title"}
        if meta_t and len(meta_t) >= 12 and meta_t.lower() not in bad_meta:
            return meta_t.strip()[:800], "metadata", warnings

        if len(doc) < 1:
            return "", "none", warnings + ["title_no_pages"]

        blob = (doc.load_page(0).get_text("text") or "").strip()
        if len(blob) < 30:
            return "", "none", warnings + ["title_page_short"]

        abstract_m = re.search(r"(?im)^\s*abstract\s*$", blob)
        cut = blob[: abstract_m.start()] if abstract_m else blob

        paras = [p.strip() for p in re.split(r"\n\s*\n", cut) if p.strip()]
        for p in paras:
            if len(p) < 30:
                continue
            if len(p) > 700:
                continue
            lows = p.lower()
            if lows.startswith(("received ", "accepted ", "published ", "copyright")):
                continue
            if "@" in p:
                continue
            if len(p.split()) < 4:
                continue
            return " ".join(p.split())[:800], "pdf_heuristic", warnings

        lines = [ln.strip() for ln in cut.splitlines() if ln.strip()]
        lines = [
            ln
            for ln in lines
            if len(ln) >= 12 and not re.match(r"^(?:page\s+\d+|\d+)\s*$", ln, re.I)
        ]
        joined_parts: list[str] = []
        for ln in lines[:12]:
            if _COPYRIGHTISH.search(ln):
                break
            if ln.count("@") >= 2:
                break
            if re.match(r"(?i)^(introduction|keywords)\s*$", ln):
                break
            joined_parts.append(ln)
            cand = " ".join(joined_parts)
            if 45 <= len(cand) <= 700 and len(joined_parts) >= 2:
                return cand[:800], "pdf_heuristic", warnings + ["title_multiline_concat"]
            if len(cand) > 720:
                break

        single = lines[0] if lines else ""
        if single and 30 <= len(single) <= 300 and len(single.split()) >= 4:
            return single[:800], "pdf_heuristic", warnings + ["title_first_long_line"]

        return "", "none", warnings + ["title_not_detected"]
    finally:
        doc.close()


def attach_pubmed_enrichment(record: dict, enrichment: dict) -> dict:
    """Attach NCBI E-utilities payload; mutates ``record`` and returns it."""
    record["pubmed_enrichment"] = enrichment
    return record


def _flush_mupdf_messages(pdf_path: str, abstract_json_path_str: str, warnings_out: list[str]) -> None:
    """
    Drain MuPDF's warning buffer: append ``mupdf:…`` to ``warnings_out`` and print stderr
    lines that start with ``ABSTRACT_MUPDF_WARN:`` (PDF path only) so logs are easy to grep;
    JSON path and deduplicated message fragments follow on separate lines.
    """
    blob = (fitz.TOOLS.mupdf_warnings(reset=True) or "").strip()
    if not blob:
        return
    one_line = " | ".join(ln.strip() for ln in blob.splitlines() if ln.strip())
    warnings_out.append(f"mupdf:{one_line}")

    # Blank line separates tqdm's \r-terminated stderr noise from grep-friendly anchors.
    print("", file=sys.stderr, flush=True)
    print(f"ABSTRACT_MUPDF_WARN: {pdf_path}", file=sys.stderr, flush=True)
    print(f"ABSTRACT_MUPDF_WARN_JSON: {abstract_json_path_str}", file=sys.stderr, flush=True)

    fragments = [p.strip() for p in one_line.split("|") if p.strip()]
    uniq: list[str] = []
    seen: set[str] = set()
    for frag in fragments:
        if frag not in seen:
            seen.add(frag)
            uniq.append(frag)

    print("ABSTRACT_MUPDF_WARN_DETAIL:", file=sys.stderr, flush=True)
    detail_limit = 50
    for frag in uniq[:detail_limit]:
        print(f"  {frag}", file=sys.stderr, flush=True)
    if len(uniq) > detail_limit:
        omitted = len(uniq) - detail_limit
        print(f"  … ({omitted} more distinct message(s) omitted)", file=sys.stderr, flush=True)


def extract_record_for_pdf(
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None = None,
) -> dict:
    """Build full JSON-serializable record for one PDF."""
    warnings: list[str] = []
    pdf_p = Path(pdf_path)
    fitz.TOOLS.reset_mupdf_warnings()
    abstract_meta_path = abstract_json_path(
        pdf_path, papers_dir, abstract_meta_root=abstract_meta_root
    ).resolve()
    try:
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

        title_pdf, title_src, tw = extract_document_title(str(pdf_p))
        warnings.extend(tw)

        doi_bundle = extract_dois_for_pdf_path(str(pdf_p), title_pdf)
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

        rec = {
            **doi_bundle,
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
            "title_pdf": title_pdf,
            "title_source": title_src,
            "abstract_pubmed": "",
            "warnings": warnings,
        }
        return rec
    finally:
        path_for_log = str(pdf_p.resolve())
        _flush_mupdf_messages(path_for_log, str(abstract_meta_path), warnings)


def save_abstract_record(
    record: dict,
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None = None,
) -> Path:
    """Write ``record`` pretty-printed to the sidescar JSON path for ``pdf_path``; create parent dirs."""
    path = abstract_json_path(pdf_path, papers_dir, abstract_meta_root=abstract_meta_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def load_abstract_record(
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None = None,
) -> dict | None:
    """Load sidescar JSON if it exists and parses; return ``None`` on missing file or decode error."""
    path = abstract_json_path(pdf_path, papers_dir, abstract_meta_root=abstract_meta_root)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_and_save_pdf(
    pdf_path: str,
    papers_dir: str,
    *,
    abstract_meta_root: Path | None = None,
) -> dict:
    """Convenience: extract one PDF to a record and persist; returns the in-memory dict."""
    rec = extract_record_for_pdf(
        pdf_path, papers_dir, abstract_meta_root=abstract_meta_root
    )
    save_abstract_record(
        rec,
        pdf_path,
        papers_dir,
        abstract_meta_root=abstract_meta_root,
    )
    return rec
