"""
app.py — Papers RAG: Semantic Search & Deep Chat over your PDF library.

Run with:
    conda activate papers_rag
    streamlit run app.py
"""

import re
import shutil
import time
import os
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
import html
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException
from fastembed import TextEmbedding

from indexer import (
    MAX_BOOLEAN_CLAUSES,
    clause_search,
    format_boolean_expression_preview,
    format_boolean_expression_translation_md,
    merge_chunks_for_papers,
    papers_from_hits,
    retrieve_boolean_evidence,
    get_index_stats,
    get_indexed_papers,
    index_papers,
    is_indexed,
    DB_PATH,
    PAPERS_DIR,
    EMBEDDING_MODEL,
    get_abstract_meta_stats,
    get_database_health_details,
    get_keyword_index_stats,
    get_metadata_index_stats,
    get_pdf_vector_sync_stats,
    neighboring_chunks_for_hit,
    rebuild_keyword_index,
    rebuild_paper_metadata_index,
)
from abstract_extraction import (
    ABSTRACT_META_ROOT,
    load_abstract_record,
)
from extract_abstracts import run_abstract_extractions
from papers_rag_config import (
    DATABASE_HEALTH_REPORTS_DIR,
    EXPORTED_PROMPTS_DIR,
    INDEX_DIR_NAME,
    SELECTED_PDFS_BASE,
)
from rag_engine import (
    build_external_llm_context_text,
    get_gemini_client,
    upload_pdfs_to_gcs,
    stream_pdf_chat,
    stream_rag_response,
)
from pdf_server import start_pdf_server, pdf_url, PDF_SERVER_PORT


def _streamlit_supports_fragment() -> bool:
    """Partial reruns for checkbox tweaks require Streamlit 1.37+ (st.fragment)."""
    if not hasattr(st, "fragment"):
        return False
    try:
        parts = st.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (1, 37)
    except (ValueError, IndexError):
        return True


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Papers RAG V2.5",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLAUSE_DISCOVERY_CACHE_MAX_ENTRIES = 32
APP_ROOT = Path(__file__).resolve().parent
LIBRARY_STATE_PATH = APP_ROOT / ".papers_rag_state.json"


def _derive_library_paths(papers_dir: str | Path) -> dict[str, Path]:
    """Derive all per-library working paths from one papers root."""
    papers_root = Path(papers_dir).expanduser().resolve()
    index_root = papers_root / INDEX_DIR_NAME
    return {
        "papers_dir": papers_root,
        "index_root": index_root,
        "chroma_db_path": index_root / "chroma_db",
        "abstract_meta_root": index_root / "abstract_meta",
        "exported_prompts_dir": index_root / "exported_prompts",
        "selected_pdfs_dir": index_root / "selected_pdfs",
        "health_reports_dir": index_root / "databases_health_reports",
    }


def _read_library_state() -> dict:
    try:
        with LIBRARY_STATE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_library_state(papers_dir: str | Path) -> None:
    data = {"papers_dir": str(Path(papers_dir).expanduser().resolve())}
    LIBRARY_STATE_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _active_library_paths() -> dict[str, Path]:
    state = _read_library_state()
    state_papers_dir = str(state.get("papers_dir") or "").strip()
    if state_papers_dir and Path(state_papers_dir).expanduser().is_dir():
        return _derive_library_paths(state_papers_dir)
    return _derive_library_paths(PAPERS_DIR)


_LIBRARY = _active_library_paths()
for _path in _LIBRARY.values():
    if _path == _LIBRARY["papers_dir"]:
        continue
    _path.mkdir(parents=True, exist_ok=True)

# Runtime-active library paths. These shadow startup defaults imported from
# config/indexer so Streamlit reruns can switch libraries without a process restart.
PAPERS_DIR = str(_LIBRARY["papers_dir"])
DB_PATH = str(_LIBRARY["chroma_db_path"])
ABSTRACT_META_ROOT = _LIBRARY["abstract_meta_root"]
EXPORTED_PROMPTS_DIR = _LIBRARY["exported_prompts_dir"]
SELECTED_PDFS_BASE = _LIBRARY["selected_pdfs_dir"]
DATABASE_HEALTH_REPORTS_DIR = _LIBRARY["health_reports_dir"]

# ── Start PDF file server (once per process) ──────────────────────────────────

@st.cache_resource
def _start_file_server(papers_dir: str):
    """Start ``pdf_server`` once per Streamlit process (cached) for ``localhost`` PDF links."""
    start_pdf_server(papers_dir=papers_dir, port=PDF_SERVER_PORT)
    return True

_start_file_server(PAPERS_DIR)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model() -> TextEmbedding:
    """Return the cached fastembed ``TextEmbedding`` instance (``EMBEDDING_MODEL``)."""
    return TextEmbedding(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to Gemini…")
def load_gemini_client():
    """Return a cached Vertex ``genai.Client``; ``st.stop()`` if connection or env setup fails."""
    try:
        return get_gemini_client()
    except Exception as e:
        st.error(f"Gemini connection failed: {e}")
        st.stop()


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    """Initialize missing ``st.session_state`` keys for search, boolean clauses, chats, and index cache."""
    defaults = {
        "search_results": [],       # all retrieved hits (unfiltered)
        "last_search_query": "",
        "last_search_cutoff": 0.6,
        "last_search_max_results": None,  # int | None — cap from last successful search
        "last_search_max_results_str": "",
        "manual_extra_fps": [],     # pasted / added paths not requiring a search
        "bool_clause_count": 1,
        "bool_group_splits": [],    # clause indices: new group starts after this clause
        "quick_chat_history": [],   # [{role, content}] for Quick Chat (excerpt-based)
        "deep_chat_papers": [],     # file_paths loaded into Deep Chat
        "deep_chat_history": [],    # [{role, content}] for Deep Chat
        "deep_chat_stage_dir": "",  # latest SELECTED_PDFS_BASE/<timestamp>/ path
        "gemini_uploads": {},       # {file_path: gs://URI or None} from GCS
        "indexing_done": False,
        "last_database_sync": None,
        "_index_stats_cache_gen": 0,
        "_keyword_index_stats_cache_gen": 0,
        "_metadata_index_stats_cache_gen": 0,
        "_abstract_cache_gen": 0,
        "clause_discovery_cache": OrderedDict(),
        "semantic_discovery_source": "Union: full-text chunks OR .json metadata",
        "last_boolean_search_elapsed_s": None,
        "last_boolean_search_diagnostics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sanitize_bool_splits(n_clauses: int, splits: list) -> list[int]:
    """Keep only valid split indices (boundary after clause ``s``); ``0 <= s <= n-2``."""
    if n_clauses < 2:
        return []
    max_s = n_clauses - 2
    try:
        out = sorted({int(s) for s in splits if 0 <= int(s) <= max_s})
    except (TypeError, ValueError):
        return []
    return out


def _normalize_paste_line(raw_line: str) -> str:
    """Strip markdown bullets and wrapping backticks from pasted basename/path lines."""
    s = raw_line.strip()
    if not s:
        return ""
    s = re.sub(r"^\d+\.\s+", "", s)
    s = re.sub(r"^[\-\*]\s+", "", s)
    return s.strip("`").strip()


def _compute_filtered_papers(
    raw_results: list,
    cutoff: float,
    max_results: int | None,
) -> tuple[list[tuple[str, list]], dict[str, list]]:
    """Same filtering/sorting as Tab 1 hit list; returns (sorted_papers, papers_map)."""
    papers_map: dict[str, list] = defaultdict(list)
    if not raw_results:
        return [], papers_map
    filtered = [
        h for h in raw_results
        if h.get("match_type") == "keyword" or h["score"] >= cutoff
    ]
    for hit in filtered:
        papers_map[hit["metadata"]["file_path"]].append(hit)
    sorted_papers = sorted(
        papers_map.items(),
        key=lambda kv: max(h["score"] for h in kv[1]),
        reverse=True,
    )
    if max_results:
        sorted_papers = sorted_papers[:max_results]
    return sorted_papers, papers_map


def _fold_paper_sets_for_ui(
    paper_sets: list[set[str]],
    split_after: list[int],
    edge_ops: list[str],
) -> set[str]:
    """Mirror ``indexer.boolean_retrieval_segmented`` paper-set combination for diagnostics."""
    if not paper_sets:
        return set()

    splits_sorted = sorted({s for s in split_after if 0 <= s <= len(paper_sets) - 2})
    ranges: list[tuple[int, int]] = []
    start = 0
    for s in splits_sorted:
        ranges.append((start, s))
        start = s + 1
    ranges.append((start, len(paper_sets) - 1))

    def _apply(left: set[str], op: str, right: set[str]) -> set[str]:
        op_u = (op or "AND").upper()
        if op_u == "AND":
            return left & right
        if op_u == "OR":
            return left | right
        if op_u == "NOT":
            return left - right
        return left & right

    seg_sets: list[set[str]] = []
    for lo, hi in ranges:
        acc = set(paper_sets[lo])
        for j in range(lo, hi):
            acc = _apply(acc, edge_ops[j], paper_sets[j + 1])
        seg_sets.append(acc)

    acc = set(seg_sets[0])
    for gi in range(1, len(seg_sets)):
        acc = _apply(acc, edge_ops[splits_sorted[gi - 1]], seg_sets[gi])
    return acc


def _clear_clause_discovery_cache() -> None:
    """Clear session-scoped Boolean clause discovery cache."""
    st.session_state["clause_discovery_cache"] = OrderedDict()


def _clause_discovery_cache() -> OrderedDict:
    """Return the bounded session LRU cache for clause discovery hits."""
    cache = st.session_state.get("clause_discovery_cache")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache or {})
        st.session_state["clause_discovery_cache"] = cache
    return cache


def _clause_cache_index_signature() -> tuple[int, int, int, int, int]:
    """Small signature for invalidating session clause cache after index changes."""
    gen = int(st.session_state.get("_index_stats_cache_gen", 0))
    stats = _cached_index_stats(gen, DB_PATH)
    meta_gen = int(st.session_state.get("_metadata_index_stats_cache_gen", 0))
    meta_stats = _cached_metadata_index_stats(meta_gen, DB_PATH)
    return (
        gen,
        int(stats.get("total_chunks") or 0),
        int(stats.get("total_papers") or 0),
        meta_gen,
        int(meta_stats.get("total_metadata_papers") or 0),
    )


def _clause_discovery_cache_key(
    mode: str,
    text: str,
    min_similarity: float,
    n_results: int,
    semantic_source: str,
) -> tuple:
    """Cache key for one clause discovery result."""
    return (
        mode,
        text.strip(),
        round(float(min_similarity), 4),
        int(n_results),
        semantic_source,
        EMBEDDING_MODEL,
        DB_PATH,
        _clause_cache_index_signature(),
        os.getenv("PAPER_DISCOVERY_BATCH_CHUNKS", "5000"),
        os.getenv("PAPER_DISCOVERY_MAX_CHUNKS", "20000"),
        os.getenv("KEYWORD_SEARCH_BACKEND", "fts"),
        os.getenv("KEYWORD_MAX_PAPERS", "0"),
    )


def _get_cached_clause_discovery(key: tuple) -> list[dict] | None:
    """Return cached clause hits and mark the entry recently used."""
    cache = _clause_discovery_cache()
    if key not in cache:
        return None
    hits = cache.pop(key)
    cache[key] = hits
    return list(hits)


def _put_cached_clause_discovery(key: tuple, hits: list[dict]) -> None:
    """Store one clause discovery result, enforcing LRU size."""
    cache = _clause_discovery_cache()
    cache[key] = list(hits)
    while len(cache) > CLAUSE_DISCOVERY_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def _timed_boolean_retrieval_for_ui(
    clauses: list[dict],
    split_after: list[int],
    edge_ops: list[str],
    embed_model,
    min_similarity: float,
    semantic_source: str,
) -> tuple[list[dict], set[str], dict]:
    """Run Boolean retrieval with per-clause timings for UI diagnostics."""
    duplicate_cache: dict[tuple[str, str], list[dict]] = {}
    per_clause_hits: list[list[dict]] = []
    paper_sets: list[set[str]] = []
    clause_rows: list[dict] = []

    total_started = time.perf_counter()
    discovery_started = total_started
    for i, clause in enumerate(clauses, start=1):
        text = (clause.get("text") or "").strip()
        mode = clause.get("mode", "semantic")
        if mode not in ("semantic", "keyword"):
            mode = "semantic"
        duplicate_key = (mode, text)
        session_key = _clause_discovery_cache_key(
            mode, text, min_similarity, 500, semantic_source
        )
        cache_source = ""
        reused = duplicate_key in duplicate_cache
        if reused:
            hits = list(duplicate_cache[duplicate_key])
            elapsed = 0.0
        else:
            cached_hits = _get_cached_clause_discovery(session_key)
            if cached_hits is not None:
                hits = cached_hits
                elapsed = 0.0
                cache_source = "session"
            else:
                clause_started = time.perf_counter()
                hits = clause_search(
                    text,
                    mode,
                    embed_model,
                    DB_PATH,
                    n_results=500,
                    paper_aware=True,
                    min_similarity=min_similarity,
                    semantic_source=semantic_source,
                )
                elapsed = time.perf_counter() - clause_started
                _put_cached_clause_discovery(session_key, hits)
            duplicate_cache[duplicate_key] = list(hits)

        hit_copy = list(hits)
        papers = papers_from_hits(hit_copy)
        per_clause_hits.append(hit_copy)
        paper_sets.append(papers)
        clause_rows.append({
            "clause": i,
            "mode": mode,
            "text": text,
            "elapsed_s": elapsed,
            "reused": reused,
            "cache_source": cache_source,
            "chunk_hits": len(hit_copy),
            "paper_hits": len(papers),
            "ranked_chunks_considered": max(
                (int(h.get("discovery_ranked_chunks_considered", len(hit_copy))) for h in hit_copy),
                default=len(hit_copy),
            ),
            "selected_chunks": max(
                (int(h.get("discovery_selected_chunks", len(hit_copy))) for h in hit_copy),
                default=len(hit_copy),
            ),
            "chunks_scanned": max(
                (int(h.get("discovery_chunks_scanned", len(hit_copy))) for h in hit_copy),
                default=len(hit_copy),
            ),
            "ceiling_reached": any(
                bool(h.get("discovery_chunk_ceiling_reached")) for h in hit_copy
            ),
            "cutoff_boundary_reached": any(
                bool(h.get("discovery_cutoff_boundary_reached")) for h in hit_copy
            ),
            "expanded_to_top_n": max(
                (int(h.get("discovery_expanded_to_top_n", len(hit_copy))) for h in hit_copy),
                default=len(hit_copy),
            ),
            "batch_size": max(
                (int(h.get("discovery_batch_size", 0)) for h in hit_copy),
                default=0,
            ),
            "fetch_steps": next(
                (
                    list(h.get("discovery_fetch_steps") or [])
                    for h in hit_copy
                    if h.get("discovery_fetch_steps")
                ),
                [],
            ),
            "min_similarity": min_similarity,
            "semantic_source": semantic_source,
            "metadata_papers": max(
                (int(h.get("discovery_metadata_papers", 0)) for h in hit_copy),
                default=0,
            ),
            "fulltext_papers": max(
                (int(h.get("discovery_fulltext_papers", 0)) for h in hit_copy),
                default=0,
            ),
            "combined_papers": max(
                (int(h.get("discovery_combined_papers", 0)) for h in hit_copy),
                default=0,
            ),
            "fulltext_representative_chunks": max(
                (int(h.get("discovery_fulltext_representative_chunks", 0)) for h in hit_copy),
                default=0,
            ),
        })
    discovery_elapsed = time.perf_counter() - discovery_started

    combine_started = time.perf_counter()
    final_fps = _fold_paper_sets_for_ui(paper_sets, split_after, edge_ops)
    discovery_hits = merge_chunks_for_papers(per_clause_hits, final_fps)
    combine_elapsed = time.perf_counter() - combine_started

    evidence_started = time.perf_counter()
    evidence_hits = retrieve_boolean_evidence(clauses, final_fps, embed_model, DB_PATH)
    evidence_elapsed = time.perf_counter() - evidence_started
    merged = merge_chunks_for_papers([discovery_hits, evidence_hits], final_fps)
    if not merged:
        merged = discovery_hits

    total_elapsed = time.perf_counter() - total_started

    diagnostics = {
        "total_elapsed_s": total_elapsed,
        "discovery_elapsed_s": discovery_elapsed,
        "combine_elapsed_s": combine_elapsed,
        "evidence_elapsed_s": evidence_elapsed,
        "discovery_chunk_hits": len(discovery_hits),
        "evidence_chunk_hits": len(evidence_hits),
        "returned_chunk_hits": len(merged),
        "clauses": clause_rows,
        "duplicate_count": sum(1 for row in clause_rows if row["reused"]),
        "session_cache_hits": sum(
            1 for row in clause_rows if row.get("cache_source") == "session"
        ),
        "session_cache_size": len(_clause_discovery_cache()),
    }
    return merged, final_fps, diagnostics


def _stage_deep_chat_pdfs(source_pdf_paths: list[str]) -> Path:
    """
    Stage PDFs under ``SELECTED_PDFS_DIR/<YYYYMMDD_HHMMSS>/`` via symlinks when possible;

    Copies if symlinks fail. Returns the staging directory path (absolute).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = (SELECTED_PDFS_BASE / ts).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    for fp in source_pdf_paths:
        dest = staging / Path(fp).name
        src = Path(fp).expanduser().resolve()
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src)
        except OSError:
            shutil.copy2(src, dest)
    return staging


def _health_report_dir() -> Path:
    """Local folder for timestamped database health reports."""
    path = DATABASE_HEALTH_REPORTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _basename_list(paths: list[str]) -> str:
    return "\n".join(f"- {Path(p).name}\n  {p}" for p in paths) or "- None"


def _html_pdf_link(path: str) -> str:
    url = pdf_url(path, PAPERS_DIR)
    label = html.escape(Path(path).name)
    if url:
        return f'<a href="{html.escape(url)}" target="_blank">{label}</a>'
    return label


def _html_path_list(paths: list[str], link_pdfs: bool = False) -> str:
    if not paths:
        return "<p>None</p>"
    items = []
    for path in paths:
        label = _html_pdf_link(path) if link_pdfs else html.escape(Path(path).name)
        items.append(f"<li>{label}<br><code>{html.escape(path)}</code></li>")
    return "<ul>" + "\n".join(items) + "</ul>"


def _write_database_health_reports(details: dict, timestamp: str | None = None) -> tuple[Path, Path]:
    """Write TXT and standalone HTML reports from database health details."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _health_report_dir()
    txt_path = out_dir / f"database_health_{stamp}.txt"
    html_path = out_dir / f"database_health_{stamp}.html"

    pdf_db = details["pdf_vector_database"]
    keyword = details["keyword_index"]
    abstract = details["abstract_meta"]
    metadata = details["metadata_vector_database"]

    txt_lines = [
        "Papers RAG database health report",
        f"Generated: {stamp}",
        "",
        "PDF folder vs PDF vector database",
        f"Current PDF folder count: {details['pdf_folder']['count']}",
        f"Indexed PDF count: {pdf_db['indexed_papers']}",
        f"PDF vector chunks: {pdf_db['chunks']}",
        "",
        "PDFs present on disk but still not indexed after update:",
        "Common reasons: unreadable PDF, dead symlink, scanned/image-only PDF with no extractable text, corrupted file, or indexing error.",
        _basename_list(pdf_db["unindexed_pdfs"]),
        "",
        "PDFs indexed in Chroma but missing from disk:",
        _basename_list(pdf_db["indexed_missing_on_disk"]),
        "",
        "PDF vector database vs keyword index",
        f"Keyword index exists: {keyword.get('exists')}",
        f"Keyword index stale: {keyword.get('stale')}",
        f"Keyword chunks: {keyword.get('keyword_chunks')}",
        f"PDF vector chunks: {keyword.get('source_chunks')}",
        f"Stored source chunks: {keyword.get('stored_source_chunks')}",
        "",
        "PDF folder vs abstract_meta JSON mirrors",
        f"JSON files: {abstract['json_total']}",
        f"JSON files linked to existing PDFs: {abstract['linked_existing']}",
        f"Unique existing PDFs represented by JSON: {abstract['unique_existing_pdfs']}",
        "",
        "PDFs without JSON:",
        _basename_list(abstract["pdfs_without_json"]),
        "",
        "JSON records pointing to missing PDFs:",
    ]
    if abstract["json_records_missing_pdf"]:
        for row in abstract["json_records_missing_pdf"]:
            txt_lines.append(f"- {row.get('file_name') or '(unknown PDF)'}")
            txt_lines.append(f"  JSON: {row.get('json_path')}")
            txt_lines.append(f"  PDF:  {row.get('file_path')}")
            if row.get("title"):
                txt_lines.append(f"  Title: {row.get('title')}")
    else:
        txt_lines.append("- None")
    txt_lines.extend([
        "",
        "Unreadable/problem JSON files:",
        _basename_list(abstract["unreadable_json"]),
        "",
        "abstract_meta JSON vs metadata vector database",
        f"JSON files linked to existing PDFs: {metadata['usable_json_records']}",
        f"Unique existing PDFs represented by JSON: {metadata['unique_existing_pdfs']}",
        f"Metadata vector records: {metadata['records']}",
        f"Metadata vector stale: {metadata['stale']}",
        "",
    ])
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    missing_json_rows = []
    for row in abstract["json_records_missing_pdf"]:
        missing_json_rows.append(
            "<li>"
            f"<strong>{html.escape(row.get('file_name') or '(unknown PDF)')}</strong>"
            f"<br>JSON: <code>{html.escape(row.get('json_path') or '')}</code>"
            f"<br>PDF: <code>{html.escape(row.get('file_path') or '')}</code>"
            + (f"<br>Title: {html.escape(row.get('title') or '')}" if row.get("title") else "")
            + "</li>"
        )
    missing_json_html = "<ul>" + "\n".join(missing_json_rows) + "</ul>" if missing_json_rows else "<p>None</p>"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Papers RAG database health report {html.escape(stamp)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; line-height: 1.45; }}
    h1, h2 {{ line-height: 1.2; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.25rem; border-radius: 3px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem; }}
    li {{ margin-bottom: 0.55rem; }}
  </style>
</head>
<body>
  <h1>Papers RAG Database Health Report</h1>
  <p>Generated: <code>{html.escape(stamp)}</code></p>
  <p>PDF links use the local PDF preview server on port <code>{PDF_SERVER_PORT}</code>; they work when the app server is running.</p>

  <div class="summary">
    <div class="card"><strong>PDF folder</strong><br>{details['pdf_folder']['count']:,} PDFs</div>
    <div class="card"><strong>PDF vector database</strong><br>{pdf_db['indexed_papers']:,} PDFs · {pdf_db['chunks']:,} chunks</div>
    <div class="card"><strong>Keyword index</strong><br>{int(keyword.get('keyword_chunks') or 0):,} chunks · stale: {html.escape(str(keyword.get('stale')))}</div>
    <div class="card"><strong>Metadata vector database</strong><br>{metadata['records']:,} records · stale: {html.escape(str(metadata['stale']))}</div>
  </div>

  <h2>PDF Folder vs PDF Vector Database</h2>
  <h3>PDFs present on disk but still not indexed after update</h3>
  <p>Common reasons: unreadable PDF, dead symlink, scanned/image-only PDF with no extractable text, corrupted file, or indexing error.</p>
  {_html_path_list(pdf_db['unindexed_pdfs'], link_pdfs=True)}
  <h3>PDFs indexed in Chroma but missing from disk</h3>
  {_html_path_list(pdf_db['indexed_missing_on_disk'], link_pdfs=False)}

  <h2>PDF Vector Database vs Keyword Index</h2>
  <ul>
    <li>Keyword index exists: <strong>{html.escape(str(keyword.get('exists')))}</strong></li>
    <li>Keyword index stale: <strong>{html.escape(str(keyword.get('stale')))}</strong></li>
    <li>Keyword chunks: <strong>{int(keyword.get('keyword_chunks') or 0):,}</strong></li>
    <li>PDF vector chunks: <strong>{int(keyword.get('source_chunks') or 0):,}</strong></li>
    <li>Stored source chunks: <strong>{int(keyword.get('stored_source_chunks') or 0):,}</strong></li>
  </ul>

  <h2>PDF Folder vs abstract_meta JSON Mirrors</h2>
  <p>{abstract['json_total']:,} JSON files · {abstract['linked_existing']:,} linked to existing PDFs · {abstract['unique_existing_pdfs']:,} unique existing PDFs</p>
  <h3>PDFs without JSON</h3>
  {_html_path_list(abstract['pdfs_without_json'], link_pdfs=True)}
  <h3>JSON records pointing to missing PDFs</h3>
  {missing_json_html}
  <h3>Unreadable/problem JSON files</h3>
  {_html_path_list(abstract['unreadable_json'], link_pdfs=False)}

  <h2>abstract_meta JSON vs Metadata Vector Database</h2>
  <ul>
    <li>JSON files linked to existing PDFs: <strong>{metadata['usable_json_records']:,}</strong></li>
    <li>Unique existing PDFs represented by JSON: <strong>{metadata['unique_existing_pdfs']:,}</strong></li>
    <li>Metadata vector records: <strong>{metadata['records']:,}</strong></li>
    <li>Metadata vector stale: <strong>{html.escape(str(metadata['stale']))}</strong></li>
  </ul>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    return txt_path, html_path


@st.cache_data(ttl=120)
def _cached_index_stats(cache_generation: int, db_path: str) -> dict:
    """Invalidate by bumping ``cache_generation`` in session state after re-indexing."""
    return get_index_stats(db_path)


@st.cache_data(ttl=120)
def _cached_metadata_index_stats(cache_generation: int, db_path: str) -> dict:
    """Invalidate by bumping ``cache_generation`` after rebuilding metadata index."""
    return get_metadata_index_stats(db_path)


@st.cache_data(ttl=120)
def _cached_keyword_index_stats(cache_generation: int, db_path: str) -> dict:
    """Invalidate by bumping ``cache_generation`` after rebuilding keyword index."""
    return get_keyword_index_stats(db_path)


@st.cache_data(ttl=120)
def _cached_pdf_vector_sync_stats(cache_generation: int, papers_dir: str, db_path: str) -> dict:
    """Compare the current PDF folder with the PDF vector database."""
    return get_pdf_vector_sync_stats(papers_dir, db_path)


@st.cache_data(ttl=120)
def _cached_abstract_meta_stats(cache_generation: int, abstract_meta_root: str, papers_dir: str) -> dict:
    """Summarize abstract_meta JSON mirrors against the current PDF folder."""
    return get_abstract_meta_stats(Path(abstract_meta_root), papers_dir)


@st.cache_data(ttl=300)
def _cached_abstract_record(
    fp: str,
    papers_dir: str,
    abstract_meta_root: str,
    cache_generation: int,
) -> dict | None:
    """Load one abstract JSON record; invalidate by bumping ``cache_generation``."""
    return load_abstract_record(fp, papers_dir, abstract_meta_root=Path(abstract_meta_root))


def _load_abstract_record(fp: str) -> dict | None:
    """Session-aware cached abstract JSON loader."""
    return _cached_abstract_record(
        fp,
        PAPERS_DIR,
        str(ABSTRACT_META_ROOT),
        int(st.session_state.get("_abstract_cache_gen", 0)),
    )


def _pubmed_esummary_from_abstract_record(rec: dict) -> dict:
    """Return PubMed esummary dict from an abstract JSON record, if present."""
    enrich = rec.get("pubmed_enrichment")
    if isinstance(enrich, dict) and isinstance(enrich.get("esummary"), dict):
        return enrich["esummary"]
    return {}


def _title_and_edge_authors_from_abstract_record(rec: dict) -> tuple[str, str, str]:
    """Return best title plus first/last author names from abstract/PubMed JSON."""
    esummary = _pubmed_esummary_from_abstract_record(rec)
    title = (
        (esummary.get("title") or "").strip()
        or (rec.get("title_pdf") or "").strip()
        or (rec.get("title_guess") or "").strip()
        or (rec.get("file_name") or "").strip()
    )
    authors = esummary.get("authors")
    names: list[str] = []
    if isinstance(authors, list):
        names = [
            (a.get("name") or "").strip()
            for a in authors
            if isinstance(a, dict) and (a.get("name") or "").strip()
        ]
    first_author = names[0] if names else ""
    last_author = names[-1] if len(names) > 1 else ""
    return title, first_author, last_author


def _render_abstract_record_header(rec: dict) -> None:
    """Render title and first/last author from abstract JSON when available."""
    title, first_author, last_author = _title_and_edge_authors_from_abstract_record(rec)
    if not any((title, first_author, last_author)):
        return
    st.markdown("###### Paper metadata")
    if title:
        st.write(f"Title: {title}")
    author_bits = []
    if first_author:
        author_bits.append(f"First author: {first_author}")
    if last_author:
        author_bits.append(f"Last author: {last_author}")
    if author_bits:
        st.write(" · ".join(author_bits))


def _normalize_abstract_for_compare(text: str) -> str:
    """Normalize abstract text for duplicate PDF/PubMed display comparison."""
    text = re.sub(r"\s+", " ", (text or "").strip()).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


_ABSTRACT_EXTRA_SECTION_RE = re.compile(
    r"(?is)\b("
    r"keywords?|key words|abbreviations?|author summary|graphical abstract|"
    r"highlights?|funding|copyright|conflict of interest|competing interests|"
    r"data availability|availability of data|supplementary information"
    r")\b\s*[:.]"
)


def _trim_abstract_extra_sections(text: str) -> str:
    """Drop common non-abstract tails such as keywords or copyright notes."""
    match = _ABSTRACT_EXTRA_SECTION_RE.search(text or "")
    if not match:
        return text or ""
    head = (text or "")[: match.start()].strip()
    return head if len(head.split()) >= 40 else (text or "")


def _abstract_overlap_ratio(shorter: str, longer: str) -> float:
    """Approximate how much of shorter is covered by longer after normalization."""
    matcher = SequenceMatcher(None, shorter, longer, autojunk=False)
    covered = sum(block.size for block in matcher.get_matching_blocks())
    return covered / max(len(shorter), 1)


def _abstract_similarity_status(left: str, right: str) -> str:
    """Classify PDF/PubMed abstracts as identical, substantially_same, or different."""
    left_norm = _normalize_abstract_for_compare(left)
    right_norm = _normalize_abstract_for_compare(right)
    if not left_norm or not right_norm:
        return "different"
    if left_norm == right_norm:
        return "identical"

    left_core = _normalize_abstract_for_compare(_trim_abstract_extra_sections(left))
    right_core = _normalize_abstract_for_compare(_trim_abstract_extra_sections(right))
    core_short, core_long = sorted((left_core, right_core), key=len)
    if len(core_short) >= 120:
        if core_short in core_long:
            return "substantially_same"
        if _abstract_overlap_ratio(core_short, core_long) >= 0.94:
            return "substantially_same"
    if SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio() >= 0.94:
        return "substantially_same"
    return "different"


def _keyword_query_to_regex(query: str) -> re.Pattern | None:
    """Compile a plain/wildcard keyword query into a text highlighter regex."""
    terms: list[str] = []
    for raw in (query or "").split():
        token = raw.strip().strip('"').strip("'")
        if not token:
            continue
        if token.endswith("*") and len(token) > 1:
            base = re.escape(token[:-1])
            terms.append(rf"\b{base}[\w-]*")
        else:
            terms.append(re.escape(token))
    if not terms:
        return None
    return re.compile("(" + "|".join(terms) + ")", flags=re.IGNORECASE)


def _normalize_excerpt_for_display(text: str) -> str:
    """Collapse PDF extraction line-break noise for readable excerpt display."""
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _highlight_keyword_text(text: str, query: str) -> tuple[str, int]:
    """Return escaped HTML with keyword query matches highlighted in red."""
    pattern = _keyword_query_to_regex(query)
    text = text or ""
    if not pattern:
        return html.escape(text), 0

    parts: list[str] = []
    last = 0
    count = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[last:match.start()]))
        parts.append(
            '<mark style="background-color:#ffe5e5; color:#b00020; '
            'font-weight:700; padding:0 2px;">'
            f"{html.escape(match.group(0))}</mark>"
        )
        last = match.end()
        count += 1
    parts.append(html.escape(text[last:]))
    return "".join(parts), count


def _render_hit_text(hit: dict) -> None:
    """Render matching excerpt text, highlighting keyword hits when available."""
    if hit.get("match_type") != "keyword":
        st.write(hit["text"])
        return
    query = hit.get("keyword_query") or ""
    if query:
        st.caption(f"keyword query: `{query}`")
    else:
        st.caption("keyword query unavailable for this result; rerun search or clear search cache.")
        st.write(hit["text"])
        return

    context_text = neighboring_chunks_for_hit(hit, DB_PATH, window=2)
    display_text = _normalize_excerpt_for_display(context_text or hit.get("text") or "")
    highlighted, match_count = _highlight_keyword_text(display_text, query)
    if match_count == 0:
        st.caption(
            "No literal keyword occurrence found in the expanded displayed excerpt. "
            "This can happen with older cached results or tokenization differences."
        )
        st.write(hit["text"])
        return
    st.caption("Showing expanded context around the keyword hit.")
    st.markdown(
        f'<div style="line-height:1.45;">{highlighted}</div>',
        unsafe_allow_html=True,
    )


def _render_abstract_texts(rec: dict) -> None:
    """Render PDF/PubMed abstracts once when they are effectively identical."""
    pdf_abs = (rec.get("abstract_text") or "").strip()
    pubmed_abs = (rec.get("abstract_pubmed") or "").strip()

    similarity_status = _abstract_similarity_status(pdf_abs, pubmed_abs)
    if pdf_abs and pubmed_abs and similarity_status in ("identical", "substantially_same"):
        if similarity_status == "identical":
            st.caption("PDF extraction and PubMed abstract are identical.")
        else:
            st.caption(
                "PDF extraction and PubMed abstract are substantially the same; "
                "showing one copy."
            )
        st.markdown("###### Abstract")
        st.write(pdf_abs)
        return

    if pdf_abs:
        st.markdown("###### Abstract (PDF extraction)")
        st.write(pdf_abs)
    if pubmed_abs:
        st.markdown("###### Abstract (PubMed / NCBI)")
        st.write(pubmed_abs)
    if not pdf_abs and not pubmed_abs:
        st.write("_(empty)_")


def _clear_manual_add_callback() -> None:
    """Clear pasted manual-add list and deselect manual-only papers (hits keep selection)."""
    prev = list(st.session_state.get("manual_extra_fps") or [])
    st.session_state["manual_extra_fps"] = []
    st.session_state["paste_apply_summary"] = ""
    if "paste_basenames" in st.session_state:
        st.session_state["paste_basenames"] = ""

    cutoff = float(
        st.session_state.get(
            "_tab1_cutoff_live",
            st.session_state.get("last_search_cutoff", 0.6),
        )
    )
    max_results = st.session_state.get("_tab1_max_results_live")
    raw = st.session_state.get("search_results", [])
    sorted_papers, _ = _compute_filtered_papers(raw, cutoff, max_results)
    hit_fps = {fp for fp, _ in sorted_papers}

    # Only drop selection for paths that were manual-only. Papers that also appear
    # in the current hit list keep ``sel_*`` so checkboxes / Export stay consistent.
    for fp in prev:
        if fp not in hit_fps:
            st.session_state[f"sel_{fp}"] = False


def _clear_tab1_all_callback() -> None:
    """Reset Tab 1 search results, manual adds, quick chat, boolean splits, and all ``sel_*`` checkboxes."""
    _clear_manual_add_callback()
    st.session_state["search_results"] = []
    st.session_state["last_search_query"] = ""
    st.session_state["quick_chat_history"] = []
    st.session_state["last_search_max_results_str"] = ""
    st.session_state["bool_group_splits"] = []
    st.session_state["last_boolean_search_elapsed_s"] = None
    st.session_state["last_boolean_search_diagnostics"] = None
    st.session_state["paste_apply_summary"] = ""
    for k in list(st.session_state.keys()):
        if k.startswith("sel_"):
            del st.session_state[k]


def _make_bulk_selection_callback(hit_fps: tuple[str, ...], manual_fps: tuple[str, ...], value: bool):
    """Build an ``on_click`` handler that sets ``sel_{fp}`` for all hit and manual paths to ``value``."""

    def _cb() -> None:
        """Set ``sel_{fp}`` session keys to ``value`` for every hit and manual path."""
        all_fps = list(dict.fromkeys([*hit_fps, *manual_fps]))
        for fp in all_fps:
            st.session_state[f"sel_{fp}"] = value

    return _cb


def _score_color(score: float) -> str:
    """Traffic-light color name for similarity scores in the hit list (green / orange / red)."""
    pct = int(score * 100)
    if pct >= 75:
        return "green"
    if pct >= 55:
        return "orange"
    return "red"


def _selection_state_label(fp: str) -> str:
    """Readable selection state for checkbox labels."""
    return "Selected" if st.session_state.get(f"sel_{fp}", False) else "Not selected"


@st.cache_data(show_spinner=False)
def _basename_resolve_map(db_path: str, cache_generation: int) -> dict[str, list[str]]:
    """casefold basename -> list of indexed absolute paths (may be ambiguous)."""
    _ = cache_generation
    d: dict[str, list[str]] = defaultdict(list)
    for p in get_indexed_papers(db_path):
        k = Path(p["file_path"]).name.casefold()
        d[k].append(p["file_path"])
    return {k: sorted(set(v)) for k, v in d.items()}


def _abstract_payload(fp: str) -> dict:
    """Load ``abstract_meta`` JSON for ``fp`` or return a small placeholder dict when missing."""
    rec = _load_abstract_record(fp)
    if rec:
        return rec
    msg = "Abstract not detected."
    return {
        "file_name": Path(fp).name,
        "abstract_text": msg,
        "status": "missing_json",
        "source": "none",
        "char_count": len(msg),
        "word_count": len(msg.split()),
    }


def _flush_deep_chat_staged_banner() -> None:
    """Show one-shot list of PDFs staged for Deep Chat (survives ``st.rerun()``)."""
    payload = st.session_state.pop("_deep_chat_staged_banner", None)
    if not payload:
        return
    names = list(payload.get("names") or [])
    n = int(payload.get("count", len(names)))
    cap = 40
    lines = "\n".join(f"- `{x}`" for x in names[:cap])
    if len(names) > cap:
        lines += f"\n\n_(…and {len(names) - cap} more)_"
    folder = (payload.get("folder") or "").strip()
    loc = f"\nStaging folder:\n```\n{folder}\n```" if folder else ""
    st.success(
        f"✅ **{n}** paper(s) staged for Deep Chat timestamp subfolder "
        "(symlinks when supported — otherwise copied).\n\n"
        "Open **💬 Deep Chat with Gemini** to upload to Gemini.\n\n"
        + (lines if lines else "_(no filenames recorded)_")
        + loc
    )


def _render_paper_selection_widgets(
    sorted_papers: list[tuple[str, list]],
    manual_only: list[str],
    papers_map: dict[str, list],
    embed_model,
    gemini_client,
) -> None:
    """Selection lists + Export / Deep Chat / Quick Chat (same st.fragment as checkboxes)."""
    has_hits = bool(sorted_papers)
    has_manual = bool(manual_only)
    if not has_hits and not has_manual:
        return

    hit_fps_t = tuple(fp for fp, _ in sorted_papers)
    man_fps_t = tuple(manual_only)
    all_fps_t = tuple(dict.fromkeys([*hit_fps_t, *man_fps_t]))
    selected_count = sum(
        1 for fp in all_fps_t if st.session_state.get(f"sel_{fp}", False)
    )
    total_count = len(all_fps_t)

    st.divider()
    st.markdown(
        "**Select papers** — Quick Chat & export use **checked** rows; "
        "**Send to Deep Chat** loads all checked PDFs."
    )
    if selected_count == total_count:
        st.success(f"All **{total_count}** paper(s) selected.")
    elif selected_count == 0:
        st.warning(f"All **{total_count}** paper(s) deselected.")
    else:
        st.info(
            f"Selection: **{selected_count} of {total_count}** paper(s) selected."
        )

    col_sa, col_da = st.columns(2)
    with col_sa:
        st.button(
            "Select all",
            key="quick_sel_all",
            use_container_width=True,
            on_click=_make_bulk_selection_callback(hit_fps_t, man_fps_t, True),
        )
    with col_da:
        st.button(
            "Deselect all",
            key="quick_desel_all",
            use_container_width=True,
            on_click=_make_bulk_selection_callback(hit_fps_t, man_fps_t, False),
        )

    if has_manual:
        nm = len(manual_only)
        with st.expander(
            f"📂 Manual add (not in current search hits) — **{nm}** paper(s)",
            expanded=False,
        ):
            st.caption("**abstract_meta** JSON only — no search excerpts.")
            show_manual_abstracts = st.checkbox(
                "Load abstracts for manual-added papers",
                value=False,
                key="show_manual_abstracts",
                help="Keeps pasted-name apply fast by not loading abstract JSON for every manual row immediately.",
            )
            for fp in sorted(manual_only, key=lambda p: Path(p).name.casefold()):
                pdf_link = pdf_url(fp, PAPERS_DIR)
                name = Path(fp).name
                st.markdown(
                    f'`manual` · <a href="{pdf_link}" target="_blank"><b>{name}</b></a>',
                    unsafe_allow_html=True,
                )
                if show_manual_abstracts:
                    ap = _abstract_payload(fp)
                    with st.expander("📋 Abstract", expanded=False):
                        _render_abstract_record_header(ap)
                        st.caption(
                            f"status: **{ap['status']}** · source: `{ap['source']}` · "
                            f"{ap.get('char_count', '—')} chars · {ap.get('word_count', '—')} words"
                        )
                        _render_abstract_texts(ap)
                st.checkbox(
                    f"{_selection_state_label(fp)} · `{name}`",
                    key=f"sel_{fp}",
                )

    if has_hits:
        nh = len(sorted_papers)
        with st.expander(
            f"📂 Search hit papers — **{nh}** paper(s)",
            expanded=False,
        ):
            for fp, hits in sorted_papers:
                best_score = max(h["score"] for h in hits)
                score_pct = int(best_score * 100)
                fname = Path(fp).name
                st.checkbox(
                    f"{_selection_state_label(fp)} · **{score_pct}% match** · `{fname}`",
                    key=f"sel_{fp}",
                )

    ordered_checked: list[str] = []
    seen_oc: set[str] = set()
    for fp, _ in sorted_papers:
        if st.session_state.get(f"sel_{fp}", False) and fp not in seen_oc:
            ordered_checked.append(fp)
            seen_oc.add(fp)
    for fp in sorted(manual_only, key=lambda p: Path(p).name.casefold()):
        if st.session_state.get(f"sel_{fp}", False) and fp not in seen_oc:
            ordered_checked.append(fp)
            seen_oc.add(fp)

    has_selection = bool(ordered_checked)
    if not has_selection:
        st.caption(
            "⚠️ **No papers checked** — use the lists above (expand if folded), "
            "then check at least one row to enable **Export** and **Deep Chat**."
        )

    exp_help = (
        "Abstract JSON + excerpts for checked papers (abstract-only if no hits)."
        if has_selection
        else "Select at least one paper above first."
    )
    deep_help = (
        "Symlinks/copies each PDF under SELECTED_PDFS_DIR/<YYYYMMDD_HHMMSS>/."
        if has_selection
        else "Select at least one paper above first."
    )

    export_clicked = st.button(
        "📄 Export context for external LLM",
        use_container_width=True,
        disabled=not has_selection,
        help=exp_help,
    )
    deep_send = st.button(
        "💬 Send selected papers to Deep Chat →",
        type="primary",
        use_container_width=True,
        disabled=not has_selection,
        help=deep_help,
    )

    if export_clicked and ordered_checked:
        export_chunks: list[dict] = []
        abstracts_export: dict[str, dict] = {}
        for fp in ordered_checked:
            abstracts_export[fp] = _abstract_payload(fp)
            if fp in papers_map:
                export_chunks.extend(papers_map[fp])
        body = build_external_llm_context_text(
            export_chunks, abstracts_export, ordered_checked
        )
        out_dir = EXPORTED_PROMPTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"prompt_context_{ts}.txt"
        out_path.write_text(body, encoding="utf-8")
        st.success(f"Saved **`{out_path}`** ({len(ordered_checked)} paper(s)).")

    if deep_send and ordered_checked:
        st.session_state["deep_chat_papers"] = ordered_checked
        st.session_state["deep_chat_history"] = []
        st.session_state["gemini_uploads"] = {}

        staged_dir = _stage_deep_chat_pdfs(ordered_checked)
        folder_s = str(staged_dir)
        st.session_state["deep_chat_stage_dir"] = folder_s

        st.session_state["_deep_chat_staged_banner"] = {
            "count": len(ordered_checked),
            "names": [Path(fp).name for fp in ordered_checked],
            "folder": folder_s,
        }
        st.rerun()  # fragment-only runs skip ``tab_chat``; full rerun refreshes Tab 2.

    st.divider()
    st.subheader("💬 Quick Chat (Vertex)")
    st.caption(
        "Checked papers only: **abstract JSON** for each; **best excerpt** when the "
        "paper came from search hits. Filename citations required."
    )

    quick_history: list[dict] = st.session_state.get("quick_chat_history", [])
    quick_input = st.chat_input(
        "Ask about the checked papers…",
        key="quick_chat_input",
    )

    if quick_history:
        _, col_qcbtn = st.columns([4, 1])
        with col_qcbtn:
            if st.button("🗑 Clear", key="clear_quick_chat", use_container_width=True):
                st.session_state["quick_chat_history"] = []
                try:
                    st.rerun(scope="fragment")
                except StreamlitAPIException:
                    st.rerun()

    for msg in quick_history:
        role_display = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role_display):
            st.markdown(msg["content"])

    if quick_input:
        if not ordered_checked:
            with st.chat_message("assistant"):
                st.error("Select at least one paper before using Quick Chat.")
        else:
            quick_context_chunks: list[dict] = []
            for fp in ordered_checked:
                if fp in papers_map:
                    hlist = papers_map[fp]
                    quick_context_chunks.append(max(hlist, key=lambda h: h["score"]))

            abstracts_quick: dict[str, dict] = {}
            for fp in ordered_checked:
                abstracts_quick[fp] = _abstract_payload(fp)

            with st.chat_message("user"):
                st.markdown(quick_input)

            st.session_state["quick_chat_history"].append(
                {"role": "user", "content": quick_input}
            )

            with st.chat_message("assistant"):
                placeholder_qc = st.empty()
                full_qc_response = ""
                prior_qc_history = st.session_state["quick_chat_history"][:-1]

                try:
                    for text_chunk, sources in stream_rag_response(
                        query=quick_input,
                        chat_history=prior_qc_history,
                        embedding_model=embed_model,
                        gemini_client=gemini_client,
                        preloaded_chunks=quick_context_chunks if quick_context_chunks else [],
                        abstracts_by_file_path=abstracts_quick,
                    ):
                        if sources is not None:
                            continue
                        if text_chunk:
                            full_qc_response += text_chunk
                            placeholder_qc.markdown(full_qc_response + "▌")

                    placeholder_qc.markdown(full_qc_response)

                except Exception as e:
                    full_qc_response = f"⚠️ Error: {e}"
                    placeholder_qc.error(full_qc_response)

            st.session_state["quick_chat_history"].append(
                {"role": "model", "content": full_qc_response}
            )


ABS_SCOPE_INCREMENTAL = "Incremental: missing JSON, newer PDFs, or missing PubMed"
ABS_SCOPE_FULL_REFRESH = "Full refresh: rebuild all JSON mirrors"


def _abs_scope_on_change() -> None:
    """Clear the PDF-vs-JSON refresh flag when switching to full refresh."""
    scope = str(st.session_state.get("sidebar_abs_scope", "")).strip()
    if scope == ABS_SCOPE_FULL_REFRESH:
        st.session_state["sidebar_abs_refresh_newer_pdf"] = False


def _pubmed_credentials_configured() -> bool:
    """Return True when an Entrez contact email is available for PubMed calls."""
    return bool((os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL") or "").strip())


def _render_synchronize_databases_control() -> None:
    """Render the one-click database synchronization workflow."""
    st.subheader("Synchronize Databases")
    st.caption(
        "Recommended after adding, deleting, or renaming PDFs. Runs the update "
        "chain in order and writes TXT and HTML health reports."
    )
    sync_clicked = st.button(
        "🧭 Synchronize/Build/Update Databases",
        use_container_width=True,
    )
    last_sync = st.session_state.get("last_database_sync")
    if isinstance(last_sync, dict):
        st.success(
            f"Last sync finished in **{float(last_sync.get('elapsed_s', 0.0)):.1f}s**"
        )
        if last_sync.get("pubmed_missing_credentials"):
            st.warning(
                "PubMed enrichment was enabled but skipped because NCBI_EMAIL / "
                "ENTREZ_EMAIL is not configured. Local PDF abstract extraction still ran."
            )
        html_path = Path(last_sync.get("html_path", ""))
        txt_path = Path(last_sync.get("txt_path", ""))
        if html_path:
            try:
                html_uri = html_path.resolve().as_uri()
                st.markdown(
                    f"[HTML health report]({html_uri})",
                    unsafe_allow_html=False,
                )
            except ValueError:
                pass
        if txt_path:
            try:
                txt_uri = txt_path.resolve().as_uri()
                st.markdown(
                    f"[TXT health report]({txt_uri})",
                    unsafe_allow_html=False,
                )
            except ValueError:
                pass
        st.caption(
            "If links do not open automatically, copy the paths and open in a browser tab."
        )

    if not sync_clicked:
        return

    sync_started = time.perf_counter()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _sync_progress(start: float, end: float):
        def _inner(frac, msg):
            progress_bar.progress(start + (end - start) * float(frac))
            status_text.caption(msg)
        return _inner

    embed_model = load_embedding_model()
    use_pubmed = bool(st.session_state.get("sidebar_abs_pubmed", False))
    pubmed_missing_credentials = use_pubmed and not _pubmed_credentials_configured()

    with st.spinner("Synchronizing databases…"):
        status_text.caption("1/5 Updating PDF vector database…")
        pdf_result = index_papers(
            papers_dir=PAPERS_DIR,
            db_path=DB_PATH,
            embedding_model=embed_model,
            progress_callback=_sync_progress(0.00, 0.35),
        )
        progress_bar.progress(0.35)

        status_text.caption("2/5 Rebuilding keyword search index…")
        keyword_result = rebuild_keyword_index(
            db_path=DB_PATH,
            progress_callback=_sync_progress(0.35, 0.55),
        )
        progress_bar.progress(0.55)

        status_text.caption("3/5 Extracting missing/stale abstract_meta JSON mirrors…")
        abs_result = run_abstract_extractions(
            papers_dir=PAPERS_DIR,
            abstract_meta_root=ABSTRACT_META_ROOT,
            pubmed_meta=use_pubmed,
            force=False,
            only_missing=True,
            refresh_if_newer_pdf=True,
            refresh_if_missing_pubmed=use_pubmed,
            progress_callback=_sync_progress(0.55, 0.75),
        )
        progress_bar.progress(0.75)

        status_text.caption("4/5 Rebuilding metadata vector database…")
        metadata_result = rebuild_paper_metadata_index(
            db_path=DB_PATH,
            embedding_model=embed_model,
            abstract_meta_root=ABSTRACT_META_ROOT,
            progress_callback=_sync_progress(0.75, 0.95),
        )
        progress_bar.progress(0.95)

        status_text.caption("5/5 Checking health and writing reports…")
        details = get_database_health_details(PAPERS_DIR, DB_PATH, ABSTRACT_META_ROOT)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_report, html_report = _write_database_health_reports(details, stamp)
        progress_bar.progress(1.0)

    progress_bar.empty()
    status_text.empty()

    st.session_state["_index_stats_cache_gen"] = (
        int(st.session_state.get("_index_stats_cache_gen", 0)) + 1
    )
    st.session_state["_keyword_index_stats_cache_gen"] = (
        int(st.session_state.get("_keyword_index_stats_cache_gen", 0)) + 1
    )
    st.session_state["_abstract_cache_gen"] = (
        int(st.session_state.get("_abstract_cache_gen", 0)) + 1
    )
    st.session_state["_metadata_index_stats_cache_gen"] = (
        int(st.session_state.get("_metadata_index_stats_cache_gen", 0)) + 1
    )
    _clear_clause_discovery_cache()
    try:
        _basename_resolve_map.clear()
    except Exception:
        pass
    st.session_state["last_database_sync"] = {
        "elapsed_s": time.perf_counter() - sync_started,
        "txt_path": str(txt_report),
        "html_path": str(html_report),
        "pdf_result": pdf_result,
        "keyword_result": keyword_result,
        "abstract_result": abs_result,
        "metadata_result": metadata_result,
        "pubmed_missing_credentials": pubmed_missing_credentials,
    }
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Papers RAG V2.5")
    st.caption("Semantic Search & Deep Chat over your PDF library")
    st.divider()

    index_gen = int(st.session_state.get("_index_stats_cache_gen", 0))
    keyword_gen = int(st.session_state.get("_keyword_index_stats_cache_gen", 0))
    abstract_gen = int(st.session_state.get("_abstract_cache_gen", 0))
    metadata_gen = int(st.session_state.get("_metadata_index_stats_cache_gen", 0))
    if "sidebar_abs_pubmed" not in st.session_state:
        st.session_state["sidebar_abs_pubmed"] = True
    if "sidebar_abs_refresh_newer_pdf" not in st.session_state:
        st.session_state["sidebar_abs_refresh_newer_pdf"] = True
    if st.session_state.get("sidebar_abs_scope") in (
        None,
        "Only papers without JSON",
    ):
        st.session_state["sidebar_abs_scope"] = ABS_SCOPE_INCREMENTAL
    elif st.session_state.get("sidebar_abs_scope") == "All papers (full refresh)":
        st.session_state["sidebar_abs_scope"] = ABS_SCOPE_FULL_REFRESH

    st.subheader("Papers Folder")
    st.caption("Copy the full path to your papers folder.")
    proposed_papers_dir = st.text_input(
        "Papers root",
        value=PAPERS_DIR,
        help="Local folder containing the PDFs/articles to index.",
        key="library_root_input",
    )
    proposed_path = Path(proposed_papers_dir).expanduser().resolve()
    current_papers_path = Path(PAPERS_DIR).expanduser().resolve()
    if proposed_papers_dir.strip() and proposed_path != current_papers_path:
        if not proposed_path.is_dir():
            st.error(f"Folder does not exist: {proposed_path}")
        else:
            _write_library_state(proposed_path)
            st.session_state["last_database_sync"] = None
            st.session_state["search_results"] = []
            st.session_state["deep_chat_papers"] = []
            st.session_state["deep_chat_history"] = []
            st.session_state["_index_stats_cache_gen"] = (
                int(st.session_state.get("_index_stats_cache_gen", 0)) + 1
            )
            st.session_state["_keyword_index_stats_cache_gen"] = (
                int(st.session_state.get("_keyword_index_stats_cache_gen", 0)) + 1
            )
            st.session_state["_abstract_cache_gen"] = (
                int(st.session_state.get("_abstract_cache_gen", 0)) + 1
            )
            st.session_state["_metadata_index_stats_cache_gen"] = (
                int(st.session_state.get("_metadata_index_stats_cache_gen", 0)) + 1
            )
            _clear_clause_discovery_cache()
            st.rerun()

    st.caption(f"Index root: `{INDEX_DIR_NAME}/`")
    st.divider()

    _render_synchronize_databases_control()
    st.divider()

    # ── 1. PDF vector database ───────────────────────────────────────────────
    st.subheader("1. Full-Text PDF Vector Database")
    pdf_sync = _cached_pdf_vector_sync_stats(index_gen, PAPERS_DIR, DB_PATH)
    indexed = bool(pdf_sync.get("total_chunks", 0) > 0)
    if indexed:
        st.success(
            f"✅ PDF vector database ready\n\n"
            f"**{int(pdf_sync['indexed_papers']):,}** indexed PDFs · "
            f"**{int(pdf_sync['total_chunks']):,}** chunks"
        )
    else:
        st.warning("⚠️ PDF vector database not built yet")
    st.caption(
        f"Built from parsed PDF chunks. Used for body-text semantic search. "
        f"Current PDF folder: **{int(pdf_sync['current_pdfs']):,}** PDFs. "
        f"Model: `{EMBEDDING_MODEL}`"
    )
    if pdf_sync.get("missing_on_disk") or pdf_sync.get("unindexed_pdfs"):
        st.warning(
            "PDF folder and vector database need attention: "
            f"**{int(pdf_sync['unindexed_pdfs']):,}** PDFs present on disk but not currently indexed · "
            f"**{int(pdf_sync['missing_on_disk']):,}** indexed PDFs missing from disk. "
            "Run Synchronize Databases, or use this update button followed by the keyword index rebuild."
        )

    btn_label = (
        "🔄 Build/Update PDF Vector Database"
        if indexed
        else "⚙️ Build PDF Vector Database (first run)"
    )
    if st.button(btn_label, use_container_width=True):
        embed_model = load_embedding_model()
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress(frac, msg):
            """Forward ``index_papers`` progress into the sidebar progress bar + caption."""
            progress_bar.progress(frac)
            status_text.caption(msg)

        with st.spinner("Indexing…"):
            result = index_papers(
                papers_dir=PAPERS_DIR,
                db_path=DB_PATH,
                embedding_model=embed_model,
                progress_callback=_progress,
            )
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Done! Indexed **{result['indexed']}** · "
            f"Skipped **{result['skipped']}** · "
            f"Errors **{result['errors']}** · "
            f"Removed stale PDFs **{result.get('removed_papers', 0)}** · "
            f"Removed stale chunks **{result.get('removed_chunks', 0):,}** · "
            f"Chunks **{result['total_chunks']:,}**"
        )
        st.session_state["indexing_done"] = True
        st.session_state["_index_stats_cache_gen"] = (
            int(st.session_state.get("_index_stats_cache_gen", 0)) + 1
        )
        st.session_state["_keyword_index_stats_cache_gen"] = (
            int(st.session_state.get("_keyword_index_stats_cache_gen", 0)) + 1
        )
        _clear_clause_discovery_cache()
        try:
            _basename_resolve_map.clear()
        except Exception:
            pass
        st.rerun()

    # ── 2. Keyword search index ──────────────────────────────────────────────
    st.divider()
    st.subheader("2. Keyword Search Index")
    keyword_stats = _cached_keyword_index_stats(keyword_gen, DB_PATH)
    keyword_chunks = int(keyword_stats.get("keyword_chunks") or 0)
    source_chunks = int(keyword_stats.get("source_chunks") or 0)
    if keyword_stats.get("exists") and not keyword_stats.get("stale"):
        st.success(
            "✅ SQLite FTS (Full-Text Search) keyword index ready\n\n"
            f"**{keyword_chunks:,}** chunks"
        )
    elif keyword_stats.get("exists"):
        st.warning(
            "⚠️ SQLite FTS (Full-Text Search) keyword index is stale\n\n"
            f"Keyword chunks: **{keyword_chunks:,}** · "
            f"PDF vector chunks: **{source_chunks:,}**"
        )
    else:
        st.warning("⚠️ SQLite FTS (Full-Text Search) keyword index not built yet")
    st.caption(
        "Built from PDF vector database chunk text. Used for exact keyword and "
        "prefix searches such as `GSE*`. Keyword clauses do not use semantic similarity."
    )
    if st.button(
        "🔤 Build/Update Keyword Search Index",
        use_container_width=True,
        disabled=not indexed,
    ):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _keyword_progress(frac, msg):
            progress_bar.progress(frac)
            status_text.caption(msg)

        with st.spinner("Building keyword search index…"):
            result = rebuild_keyword_index(
                db_path=DB_PATH,
                progress_callback=_keyword_progress,
            )
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Keyword index done! Indexed **{result['indexed_chunks']:,}** chunks"
        )
        st.session_state["_keyword_index_stats_cache_gen"] = (
            int(st.session_state.get("_keyword_index_stats_cache_gen", 0)) + 1
        )
        _clear_clause_discovery_cache()
        st.rerun()

    # ── 3. JSON mirrors ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("3. abstract_meta JSON Mirrors")
    abs_stats = _cached_abstract_meta_stats(abstract_gen, str(ABSTRACT_META_ROOT), PAPERS_DIR)
    if abs_stats.get("json_total"):
        st.info(
            f"**{int(abs_stats['json_total']):,}** JSON files · "
            f"**{int(abs_stats.get('linked_existing') or 0):,}** linked to existing PDFs · "
            f"**{int(abs_stats.get('unique_existing_pdfs') or 0):,}** unique PDFs"
        )
    else:
        st.warning("⚠️ No abstract_meta JSON mirrors found")
    if abs_stats.get("linked_missing") or abs_stats.get("missing_json_for_pdfs") or abs_stats.get("unreadable"):
        st.warning(
            "JSON mirrors and PDF folder are not fully synchronized: "
            f"**{int(abs_stats['missing_json_for_pdfs']):,}** PDFs without JSON · "
            f"**{int(abs_stats['linked_missing']):,}** JSON records point to missing PDFs · "
            f"**{int(abs_stats['unreadable']):,}** unreadable/problem JSON files."
        )
    st.caption(
        "Sidecar JSON mirrors live under **ABSTRACT_META_ROOT** "
        "(**papers_rag_config** · **.env**). Incremental mode also enriches "
        "existing JSON files missing **pubmed_enrichment.status**."
    )
    _abs_mode = st.radio(
        "Scope",
        (ABS_SCOPE_INCREMENTAL, ABS_SCOPE_FULL_REFRESH),
        key="sidebar_abs_scope",
        help=(
            "Incremental mode matches Synchronize. Full refresh rebuilds sidescar "
            "JSON even when files already exist."
        ),
        on_change=_abs_scope_on_change,
    )
    _abs_pubmed = st.checkbox("Query PubMed (NCBI)", key="sidebar_abs_pubmed")
    st.caption(
        "PubMed enrichment is attempted by default when NCBI_EMAIL or ENTREZ_EMAIL is configured."
    )
    if _abs_pubmed and not _pubmed_credentials_configured():
        st.warning(
            "PubMed is enabled but NCBI_EMAIL / ENTREZ_EMAIL is not configured. "
            "Extraction will continue with local PDF abstracts only."
        )
    _abs_refresh_pdf = st.checkbox(
        "Also refresh when PDF is newer than JSON",
        key="sidebar_abs_refresh_newer_pdf",
        help=(
            "Only with incremental mode. "
            "Unchecks automatically when switching to full refresh."
        ),
        disabled=(_abs_mode != ABS_SCOPE_INCREMENTAL),
    )

    if st.button("📄 Extract / refresh abstract_meta", use_container_width=True):
        prog_abs = st.progress(0.0)
        caption_abs = st.empty()

        def _abs_progress(frac, msg):
            prog_abs.progress(frac)
            caption_abs.caption(msg)

        only_missing_abs = _abs_mode == ABS_SCOPE_INCREMENTAL
        force_abs = _abs_mode == ABS_SCOPE_FULL_REFRESH

        try:
            with st.spinner("Writing abstract_meta …"):
                stats_abs = run_abstract_extractions(
                    papers_dir=PAPERS_DIR,
                    abstract_meta_root=ABSTRACT_META_ROOT,
                    pubmed_meta=_abs_pubmed,
                    force=force_abs,
                    only_missing=only_missing_abs,
                    refresh_if_newer_pdf=_abs_refresh_pdf and only_missing_abs,
                    refresh_if_missing_pubmed=_abs_pubmed and only_missing_abs,
                    progress_callback=_abs_progress,
                )
        except Exception as ex:
            prog_abs.empty()
            caption_abs.empty()
            st.error(f"abstract_meta extraction failed: {ex}")
        else:
            prog_abs.empty()
            caption_abs.empty()
            st.session_state["_abstract_cache_gen"] = (
                int(st.session_state.get("_abstract_cache_gen", 0)) + 1
            )
            st.session_state["_metadata_index_stats_cache_gen"] = (
                int(st.session_state.get("_metadata_index_stats_cache_gen", 0)) + 1
            )
            st.success(
                f"Done. Extracted **{stats_abs['extracted']:,}** · "
                f"PubMed refreshed **{int(stats_abs.get('pubmed_refreshed') or 0):,}** · "
                f"Skipped **{stats_abs['skipped']:,}** · "
                f"Errors **{stats_abs['errors']:,}**"
            )
            if stats_abs.get("pubmed_skipped_missing_credentials"):
                st.warning(
                    "PubMed enrichment was skipped because NCBI_EMAIL / ENTREZ_EMAIL "
                    "is not configured. Local PDF abstract extraction still ran."
                )
            _clear_clause_discovery_cache()

    # ── 4. Metadata vector database ──────────────────────────────────────────
    st.divider()
    st.subheader("4. Metadata Vector Database")
    meta_stats = _cached_metadata_index_stats(metadata_gen, DB_PATH)
    metadata_paper_count = int(meta_stats.get("total_metadata_papers") or 0)
    expected_metadata_records = int(abs_stats.get("unique_existing_pdfs") or 0)
    if metadata_paper_count and metadata_paper_count == expected_metadata_records:
        st.success(
            f"✅ Metadata vector database ready\n\n"
            f"Metadata vectors: **{metadata_paper_count:,}** unique PDFs\n\n"
            f"JSON mirrors: **{int(abs_stats.get('linked_existing') or 0):,}** JSON files linked to existing PDFs"
        )
    elif metadata_paper_count:
        st.warning(
            "⚠️ Metadata vector database may be stale\n\n"
            f"Metadata vectors: **{metadata_paper_count:,}** unique PDFs · "
            f"expected unique PDFs: **{expected_metadata_records:,}**\n\n"
            f"JSON mirrors: **{int(abs_stats.get('linked_existing') or 0):,}** JSON files linked to existing PDFs"
        )
    else:
        st.warning("🧾 Metadata vector database not built yet")
    st.caption(
        f"Built from abstract_meta JSON title, authors, abstract, DOI/PubMed metadata. "
        f"Used for metadata-level semantic search. Model: `{EMBEDDING_MODEL}`"
    )
    if st.button("🧾 Build/Update Metadata Vector Database", use_container_width=True):
        embed_model = load_embedding_model()
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _metadata_progress(frac, msg):
            progress_bar.progress(frac)
            status_text.caption(msg)

        with st.spinner("Building metadata vector database…"):
            result = rebuild_paper_metadata_index(
                db_path=DB_PATH,
                embedding_model=embed_model,
                abstract_meta_root=ABSTRACT_META_ROOT,
                progress_callback=_metadata_progress,
            )
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Metadata vector database done! Indexed **{result['indexed']}** · "
            f"Errors **{result['errors']}** · Records **{result['total_records']}**"
        )
        st.session_state["_metadata_index_stats_cache_gen"] = (
            int(st.session_state.get("_metadata_index_stats_cache_gen", 0)) + 1
        )
        _clear_clause_discovery_cache()
        st.rerun()
    st.divider()
    st.subheader("Search Behavior")
    st.caption(
        "Semantic clauses can search the full-text PDF vector database, the JSON "
        "metadata vector database, their union, or their intersection. Choose that "
        "source in the Boolean Search tab. Keyword clauses always use the SQLite "
        "FTS keyword index."
    )
    st.caption(
        "The session clause cache reuses unchanged clause searches while tuning a "
        "Boolean query. Clearing it does not modify ChromaDB or the keyword index."
    )

    st.divider()
    st.subheader("5. Paths & Runtime")
    st.caption(f"Papers root: `{PAPERS_DIR}`")
    st.caption(f"Index root: `{INDEX_DIR_NAME}/`")
    st.caption(f"ChromaDB: `{INDEX_DIR_NAME}/chroma_db/`")
    st.caption(f"Metadata JSON: `{INDEX_DIR_NAME}/abstract_meta/`")
    st.caption(f"Exports: `{INDEX_DIR_NAME}/exported_prompts/`")
    st.caption(f"Deep Chat staging: `{INDEX_DIR_NAME}/selected_pdfs/`")
    st.caption(f"Health reports: `{INDEX_DIR_NAME}/databases_health_reports/`")
    st.caption(f"PDF preview server: `http://localhost:{PDF_SERVER_PORT}`")


# ── Main area ─────────────────────────────────────────────────────────────────

if not indexed:
    st.title("📚 Papers RAG V2.5")
    st.info(
        "👈 Click **Build PDF Vector Database** in the sidebar to get started.\n\n"
        "This scans all PDFs, extracts text, generates embeddings, and stores "
        "them locally. It runs **once** (~10–30 min for 833 PDFs) and updates "
        "incrementally after that."
    )
    st.stop()

embed_model   = load_embedding_model()
gemini_client = load_gemini_client()

tab_search, tab_chat = st.tabs(["🔍 Semantic Search", "💬 Deep Chat with Gemini"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Semantic Search
# ════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔍 Semantic Search")
    _flush_deep_chat_staged_banner()

    st.subheader("Paste PDF basenames")
    st.caption(
        "One filename per line (no paths required), **case-insensitive**. "
        "Resolves against the **indexed** corpus. Works **without** running a search."
    )
    with st.form("paste_basenames_form", clear_on_submit=False):
        st.text_area(
            "Paste basenames",
            height=120,
            key="paste_basenames",
            placeholder="paper_one.pdf\npaper_two.pdf",
            label_visibility="collapsed",
        )
        apply_paste = st.form_submit_button(
            "Apply pasted names",
            use_container_width=True,
        )
    pc1, pc2 = st.columns(2)
    with pc1:
        st.button(
            "Clear manual-add list",
            key="clear_manual_btn",
            use_container_width=True,
            on_click=_clear_manual_add_callback,
        )
    with pc2:
        st.button(
            "Clear all (search, paste, selections)",
            key="clear_tab1_all_btn",
            use_container_width=True,
            on_click=_clear_tab1_all_callback,
        )
    if apply_paste:
        blob = st.session_state.get("paste_basenames") or ""
        bmap = _basename_resolve_map(
            DB_PATH,
            int(st.session_state.get("_index_stats_cache_gen", 0)),
        )
        matched_new = []
        unknown: list[str] = []
        ambiguous: list[tuple[str, list[str]]] = []
        for raw_line in blob.splitlines():
            line = _normalize_paste_line(raw_line)
            if not line:
                continue
            key = Path(line).name.casefold()
            opts = bmap.get(key)
            if not opts:
                unknown.append(line)
            elif len(opts) > 1:
                ambiguous.append((line, opts))
            else:
                matched_new.append(opts[0])

        matched_new = list(dict.fromkeys(matched_new))
        cutoff_ap = float(st.session_state.get("last_search_cutoff", 0.6))
        max_ap = st.session_state.get("last_search_max_results")
        sp_ap, _ = _compute_filtered_papers(
            st.session_state.get("search_results", []),
            cutoff_ap,
            max_ap,
        )
        hit_fps_ap = {fp for fp, _ in sp_ap}
        in_hits = [fp for fp in matched_new if fp in hit_fps_ap]
        manual_side = [fp for fp in matched_new if fp not in hit_fps_ap]

        cur = list(st.session_state.get("manual_extra_fps") or [])
        cur_set = set(cur)
        for fp in matched_new:
            if fp not in cur_set:
                cur.append(fp)
                cur_set.add(fp)
        for fp in matched_new:
            st.session_state[f"sel_{fp}"] = True
        st.session_state["manual_extra_fps"] = cur

        lines_out = []
        if matched_new:
            lines_out.append(
                f"Applied **{len(matched_new)}** pasted name(s); "
                f"**{len(matched_new)}** selected."
            )
            if in_hits:
                ih_lines = "\n".join(f"- `{Path(fp).name}`" for fp in in_hits[:40])
                ih_more = (
                    f"\n\n_(…and {len(in_hits) - 40} more)_" if len(in_hits) > 40 else ""
                )
                lines_out.append(
                    f"**In current hit list** ({len(in_hits)}) — checkboxes under "
                    f"*Search hit papers*:\n\n{ih_lines}{ih_more}"
                )
            if manual_side:
                ms_lines = "\n".join(f"- `{Path(fp).name}`" for fp in manual_side[:40])
                ms_more = (
                    f"\n\n_(…and {len(manual_side) - 40} more)_"
                    if len(manual_side) > 40
                    else ""
                )
                lines_out.append(
                    f"**Manual-only** (not in current filtered hits) ({len(manual_side)}):\n\n"
                    f"{ms_lines}{ms_more}"
                )
        if unknown:
            unk_block = "\n".join(f"- `{u}`" for u in unknown[:25])
            more = f"\n\n_(…and {len(unknown) - 25} more)_" if len(unknown) > 25 else ""
            lines_out.append(f"**Unknown** (not in index):\n\n{unk_block}{more}")
        if ambiguous:
            amb_block = "\n".join(
                f"- `{name}` → {len(paths)} paths (skipped)" for name, paths in ambiguous[:15]
            )
            lines_out.append(f"**Ambiguous** basename — skipped:\n\n{amb_block}")
        if not lines_out:
            lines_out.append("No usable pasted basenames were found.")
        st.session_state["paste_apply_summary"] = "\n\n".join(lines_out)
        if lines_out:
            st.success(st.session_state["paste_apply_summary"])
    elif st.session_state.get("paste_apply_summary"):
        st.success(st.session_state["paste_apply_summary"])

    def _boolean_search_panel_fragment() -> None:
        """Clause rows, group splits, operators, cutoff, Search — optionally under ``st.fragment``."""
        st.divider()
        st.markdown("### Boolean search (**Clause** rows · groups · AND / OR / NOT)")
        st.caption(
            f"Up to **{MAX_BOOLEAN_CLAUSES}** rows (**Clause 1**, **Clause 2**, …). "
            "Optional **groups**: choose where a **new group** starts **after** a Clause; "
            "operators on those edges combine whole groups. "
            "Each row is **semantic** or **keyword**; keyword rows use token/phrase search "
            "and support `*` prefix matches such as `neuro*`. "
            "**Keyword** Clause rows bypass the similarity cutoff below."
        )

        def _incr_bc() -> None:
            """Increment boolean clause count up to ``MAX_BOOLEAN_CLAUSES``."""
            if st.session_state["bool_clause_count"] < MAX_BOOLEAN_CLAUSES:
                st.session_state["bool_clause_count"] += 1

        def _decr_bc() -> None:
            """Drop the last clause and prune invalid group-split indices."""
            if st.session_state["bool_clause_count"] > 1:
                st.session_state["bool_clause_count"] -= 1
                n = int(st.session_state["bool_clause_count"])
                st.session_state["bool_group_splits"] = _sanitize_bool_splits(
                    n, list(st.session_state.get("bool_group_splits") or [])
                )

        ac1, ac2 = st.columns(2)
        with ac1:
            st.button("Add clause", key="add_clause_btn", on_click=_incr_bc)
        with ac2:
            st.button("Remove last clause", key="remove_clause_btn", on_click=_decr_bc)

        n_bc = int(st.session_state["bool_clause_count"])
        splits_raw = list(st.session_state.get("bool_group_splits") or [])
        splits_sane = _sanitize_bool_splits(n_bc, splits_raw)
        if splits_sane != splits_raw:
            st.session_state["bool_group_splits"] = splits_sane

        if n_bc >= 2:
            st.multiselect(
                "Start a **new group** after Clause…",
                options=list(range(n_bc - 1)),
                format_func=lambda j: f"After Clause {j + 1}",
                key="bool_group_splits",
                help=(
                    "Each contiguous run of **Clause** rows is one parenthesized group "
                    "in the Boolean query. Operators on those edges combine **groups**; "
                    "other operators combine **Clause** rows **within** a group."
                ),
            )

        splits_set = set(
            _sanitize_bool_splits(n_bc, st.session_state.get("bool_group_splits") or [])
        )

        for i in range(n_bc):
            st.markdown(f"**Clause {i + 1}**")
            row1, row2 = st.columns([4, 1])
            with row1:
                st.text_input(
                    "clause text",
                    key=f"bc_{i}_text",
                    label_visibility="collapsed",
                    placeholder=f"Clause {i + 1} …",
                )
            with row2:
                st.selectbox(
                    "mode",
                    ["semantic", "keyword"],
                    key=f"bc_{i}_mode",
                    label_visibility="collapsed",
                )
            if i < n_bc - 1:
                if i in splits_set:
                    st.caption("— **Between groups** —")
                    st.selectbox(
                        f"Between groups: combine through Clause {i + 1} / Clause {i + 2}",
                        ["AND", "OR", "NOT"],
                        key=f"bc_between_{i}_op",
                    )
                else:
                    st.selectbox(
                        f"Within group: Clause {i + 1} with Clause {i + 2}",
                        ["AND", "OR", "NOT"],
                        key=f"bc_{i}_op",
                    )

        preview_clauses = [
            {
                "text": st.session_state.get(f"bc_{i}_text", ""),
                "mode": st.session_state.get(f"bc_{i}_mode", "semantic"),
            }
            for i in range(n_bc)
        ]
        preview_edges: list[str] = []
        for j in range(max(0, n_bc - 1)):
            if j in splits_set:
                preview_edges.append(st.session_state.get(f"bc_between_{j}_op", "AND"))
            else:
                preview_edges.append(st.session_state.get(f"bc_{j}_op", "AND"))
        preview_text = format_boolean_expression_preview(
            preview_clauses,
            sorted(splits_set),
            preview_edges,
        )
        st.markdown("**Boolean query**")
        st.text(preview_text if preview_text else "—")

        translation_md = format_boolean_expression_translation_md(
            preview_clauses,
            sorted(splits_set),
            preview_edges,
        )
        if translation_md:
            st.text("")
            st.markdown(
                "**Translation** — `(semantic)` / `(keyword)` · truncated text · "
                "boolean ops are colored: :green[**AND**] · :orange[**OR**] · :red[**NOT**]."
            )
            st.markdown(translation_md)

        col_cut, col_max = st.columns([1.2, 1.2])
        with col_cut:
            cutoff = st.number_input(
                "Min similarity",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("last_search_cutoff", 0.6)),
                step=0.01,
                format="%.2f",
                help="Semantic chunks below this are hidden; keyword chunks always shown.",
            )
        with col_max:
            max_str = st.text_input(
                "Max results",
                value=st.session_state.get("last_search_max_results_str", ""),
                placeholder="∞",
                help="Cap unique papers after filtering (empty = no limit).",
            )
            try:
                max_results = int(max_str) if max_str.strip() else None
            except ValueError:
                max_results = None

        semantic_source_options = [
            "Vector DB: full-text PDF chunks",
            "Vector DB: .json metadata",
            "Union: full-text chunks OR .json metadata",
            "Intersection: full-text chunks AND .json metadata",
        ]
        current_semantic_source = st.session_state.get(
            "semantic_discovery_source",
            "Union: full-text chunks OR .json metadata",
        )
        legacy_semantic_labels = {
            "Full-text chunks": "Vector DB: full-text PDF chunks",
            "Paper metadata": "Vector DB: .json metadata",
            "Both": "Union: full-text chunks OR .json metadata",
        }
        current_semantic_source = legacy_semantic_labels.get(
            current_semantic_source,
            current_semantic_source,
        )
        if current_semantic_source not in semantic_source_options:
            current_semantic_source = "Union: full-text chunks OR .json metadata"
        semantic_source_label = st.selectbox(
            "Semantic discovery source",
            semantic_source_options,
            index=semantic_source_options.index(current_semantic_source),
            help=(
                "Full-text chunks uses the existing PDF chunk vector database. "
                ".json metadata uses the title/abstract/PubMed vector database. "
                "Union broadens recall; intersection requires both sources to find the paper."
            ),
        )
        st.session_state["semantic_discovery_source"] = semantic_source_label
        semantic_source = {
            "Vector DB: full-text PDF chunks": "fulltext",
            "Vector DB: .json metadata": "metadata",
            "Union: full-text chunks OR .json metadata": "both",
            "Intersection: full-text chunks AND .json metadata": "intersection",
        }[semantic_source_label]

        search_btn = st.button("🔍 Search", type="primary", use_container_width=True, key="bool_search_btn")
        search_runtime_slot = st.empty()
        last_elapsed = st.session_state.get("last_boolean_search_elapsed_s")
        if isinstance(last_elapsed, (int, float)):
            search_runtime_slot.info(f"🏃 Search wall-clock time: **{last_elapsed:.3f}s**")
        last_diag = st.session_state.get("last_boolean_search_diagnostics")
        if isinstance(last_diag, dict):
            with st.expander("Search diagnostics", expanded=False):
                dup_count = int(last_diag.get("duplicate_count") or 0)
                session_hits = int(last_diag.get("session_cache_hits") or 0)
                session_cache_size = int(last_diag.get("session_cache_size") or 0)
                combine_s = float(last_diag.get("combine_elapsed_s") or 0.0)
                if dup_count:
                    st.caption(
                        f"Reused **{dup_count}** duplicate clause search(es). "
                        "Repeated clauses are searched once and reused within the same Boolean run."
                    )
                else:
                    st.caption("No duplicate clauses reused in the last Boolean run.")
                if session_hits:
                    st.caption(
                        f"Reused **{session_hits}** clause discovery result(s) "
                        "from the session cache."
                    )
                st.caption(
                    f"Session clause cache: **{session_cache_size}** / "
                    f"**{CLAUSE_DISCOVERY_CACHE_MAX_ENTRIES}** entries"
                )
                st.caption(
                    "Discovery tuning: "
                    f"`PAPER_DISCOVERY_BATCH_CHUNKS={os.getenv('PAPER_DISCOVERY_BATCH_CHUNKS', '5000')}` · "
                    f"`PAPER_DISCOVERY_MAX_CHUNKS={os.getenv('PAPER_DISCOVERY_MAX_CHUNKS', '20000')}` · "
                    f"`EVIDENCE_CHUNKS_PER_PAPER={os.getenv('EVIDENCE_CHUNKS_PER_PAPER', '3')}`"
                )
                if st.button("Clear search cache", key="clear_clause_cache_btn"):
                    _clear_clause_discovery_cache()
                    st.session_state["last_boolean_search_diagnostics"] = None
                    st.toast("Search cache cleared.")
                    st.rerun()
                discovery_s = float(last_diag.get("discovery_elapsed_s") or 0.0)
                evidence_s = float(last_diag.get("evidence_elapsed_s") or 0.0)
                evidence_hits = int(last_diag.get("evidence_chunk_hits") or 0)
                returned_hits = int(last_diag.get("returned_chunk_hits") or 0)
                st.caption(f"Paper discovery: **{discovery_s:.3f}s**")
                st.caption(f"Boolean set combination: **{combine_s:.3f}s**")
                st.caption(
                    f"Evidence retrieval for selected papers: **{evidence_s:.3f}s**; "
                    f"{evidence_hits} evidence chunks"
                )
                st.caption(f"Returned discovery + evidence chunks: **{returned_hits}**")
                if any(
                    row.get("mode") == "semantic" and row.get("ceiling_reached")
                    for row in last_diag.get("clauses") or []
                ):
                    st.caption(
                        "Some semantic clauses reached the discovery chunk ceiling; "
                        "see the clause-specific notes below."
                    )

                def _fulltext_clause_diagnostics(row: dict, cutoff_text: str, representative_chunks: int) -> tuple[str, list[str]]:
                    selected_chunks = int(row.get("selected_chunks") or representative_chunks or 0)
                    ranked_chunks = int(
                        row.get("ranked_chunks_considered")
                        or row.get("chunks_scanned")
                        or selected_chunks
                    )
                    expanded_to = int(row.get("expanded_to_top_n") or ranked_chunks)
                    fetch_steps = [
                        int(step)
                        for step in (row.get("fetch_steps") or [])
                        if int(step) > 0
                    ]
                    notes: list[str] = []
                    if fetch_steps:
                        expansion_detail = (
                            "automatic top-N expansion: "
                            + " -> ".join(f"{step:,}" for step in fetch_steps)
                        )
                        if len(fetch_steps) > 1:
                            notes.append(
                                "Automatically raised the Chroma request size: "
                                + " -> ".join(f"{step:,}" for step in fetch_steps)
                                + "."
                            )
                        else:
                            notes.append(
                                f"Checked the top {fetch_steps[0]:,} ranked chunks; "
                                "no larger request was needed."
                            )
                    elif expanded_to:
                        expansion_detail = f"automatic top-N expansion reached {expanded_to:,}"
                    else:
                        expansion_detail = "automatic top-N expansion not needed"
                    status_bits = [
                        f"considered top {ranked_chunks:,} ranked Chroma chunks",
                        expansion_detail,
                    ]
                    if row.get("ceiling_reached"):
                        status_bits.append("ceiling reached")
                        status_bits.append("more chunks may pass the cutoff beyond this ceiling")
                        notes.append(
                            "Reached the configured discovery ceiling; more chunks may pass the cutoff. "
                            "Increase `PAPER_DISCOVERY_MAX_CHUNKS` to search deeper at the same cutoff. "
                            "Lower the minimum similarity cutoff only if you want broader, lower-similarity matches."
                        )
                    elif row.get("cutoff_boundary_reached"):
                        status_bits.append(
                            "cutoff boundary reached; scan stopped after lower-scoring chunks"
                        )
                        notes.append("Stopped because the scan crossed below the similarity cutoff.")
                    else:
                        status_bits.append("scan stopped before the configured ceiling")
                        notes.append(
                            "Stopped before the configured ceiling because Chroma returned fewer chunks than requested."
                        )
                    detail = (
                        f"{selected_chunks:,} full-text chunks passed cutoff >= {cutoff_text} · "
                        f"{representative_chunks:,} representative full-text chunks · "
                        + " · ".join(status_bits)
                    )
                    return detail, notes

                for row in last_diag.get("clauses") or []:
                    if row.get("reused"):
                        reused = " · reused duplicate clause"
                    elif row.get("cache_source") == "session":
                        reused = " · reused from session cache"
                    else:
                        reused = ""
                    mode = row.get("mode")
                    clause_notes: list[str] = []
                    if mode == "semantic":
                        cutoff_text = f"{float(row.get('min_similarity') or 0.0):.2f}"
                        source = row.get("semantic_source", "fulltext")
                        if source == "metadata":
                            detail = (
                                f"{row['paper_hits']} metadata papers passed cutoff >= {cutoff_text}"
                            )
                        elif source in ("both", "intersection"):
                            meta_papers = int(row.get("metadata_papers") or 0)
                            fulltext_papers = int(row.get("fulltext_papers") or 0)
                            combined_papers = int(row.get("combined_papers") or row["paper_hits"])
                            fulltext_representative_chunks = int(
                                row.get("fulltext_representative_chunks") or fulltext_papers
                            )
                            chunk_detail, chunk_notes = _fulltext_clause_diagnostics(
                                row, cutoff_text, fulltext_representative_chunks
                            )
                            clause_notes.extend(chunk_notes)
                            join_label = "union" if source == "both" else "intersection"
                            detail = (
                                f"{fulltext_papers} full-text papers · "
                                f"{meta_papers} metadata papers · "
                                f"{combined_papers} {join_label} papers · "
                                f"full-text chunk discovery: {chunk_detail}"
                            )
                        else:
                            chunk_detail, chunk_notes = _fulltext_clause_diagnostics(
                                row, cutoff_text, int(row.get("chunk_hits") or 0)
                            )
                            clause_notes.extend(chunk_notes)
                            detail = f"{chunk_detail} · {row['paper_hits']:,} representative papers"
                    else:
                        detail = (
                            f"{row['chunk_hits']:,} representative chunks · "
                            f"{row['paper_hits']:,} papers · FTS-backed keyword discovery"
                        )
                    st.markdown(
                        f"- Clause {row['clause']} `({row['mode']})` "
                        f"`{row['text']}` — **{row['elapsed_s']:.3f}s**{reused}; "
                        f"{detail}"
                    )
                    for note in clause_notes:
                        if row.get("ceiling_reached"):
                            st.warning(f"Clause {row['clause']} note: {note}")
                        else:
                            st.info(f"Clause {row['clause']} note: {note}")

        st.session_state["_tab1_cutoff_live"] = cutoff
        st.session_state["_tab1_max_results_live"] = max_results

        if search_btn:
            n_bc = int(st.session_state["bool_clause_count"])
            splits_run = _sanitize_bool_splits(
                n_bc, list(st.session_state.get("bool_group_splits") or [])
            )
            splits_set_run = set(splits_run)
            texts = [st.session_state.get(f"bc_{i}_text", "").strip() for i in range(n_bc)]
            if not any(texts):
                st.warning("Enter at least one non-empty clause.")
            elif n_bc > 1 and any(not texts[i] for i in range(n_bc)):
                st.warning(
                    "When using multiple clauses, fill **every** clause row (or remove extras)."
                )
            else:
                clauses = [
                    {"text": texts[i], "mode": st.session_state.get(f"bc_{i}_mode", "semantic")}
                    for i in range(n_bc)
                ]
                edge_ops: list[str] = []
                for j in range(max(0, n_bc - 1)):
                    if j in splits_set_run:
                        edge_ops.append(st.session_state.get(f"bc_between_{j}_op", "AND"))
                    else:
                        edge_ops.append(st.session_state.get(f"bc_{j}_op", "AND"))
                with search_runtime_slot, st.spinner("🏃 Searching…"):
                    merged, _fps, search_diagnostics = _timed_boolean_retrieval_for_ui(
                        clauses,
                        splits_run,
                        edge_ops,
                        embed_model,
                        cutoff,
                        semantic_source,
                    )
                    search_elapsed = float(search_diagnostics["total_elapsed_s"])
                    label = format_boolean_expression_preview(
                        clauses,
                        splits_run,
                        edge_ops,
                    )
                st.session_state["search_results"] = merged
                st.session_state["last_search_query"] = label
                st.session_state["last_boolean_search_elapsed_s"] = search_elapsed
                st.session_state["last_boolean_search_diagnostics"] = search_diagnostics
                st.session_state["quick_chat_history"] = []
                st.session_state["search_generation"] = (
                    st.session_state.get("search_generation", 0) + 1
                )
                st.session_state["last_search_cutoff"] = cutoff
                st.session_state["last_search_max_results"] = max_results
                st.session_state["last_search_max_results_str"] = max_str.strip()
                st.rerun()

    _boolean_runner = (
        st.fragment()(_boolean_search_panel_fragment)
        if _streamlit_supports_fragment()
        else _boolean_search_panel_fragment
    )
    _boolean_runner()

    raw_results = st.session_state.get("search_results", [])
    cutoff = float(
        st.session_state.get(
            "_tab1_cutoff_live",
            st.session_state.get("last_search_cutoff", 0.6),
        )
    )
    max_results = st.session_state.get("_tab1_max_results_live")
    sorted_papers, papers_map = _compute_filtered_papers(raw_results, cutoff, max_results)

    hit_fps = {fp for fp, _ in sorted_papers}
    manual_extra = list(st.session_state.get("manual_extra_fps") or [])
    manual_only = [fp for fp in manual_extra if fp not in hit_fps]

    query_label = st.session_state.get("last_search_query") or ""

    if sorted_papers:
        pct = int(cutoff * 100)
        ql = query_label or "(no Boolean query label yet)"
        with st.expander(
            f"📂 Search results — **{len(sorted_papers)}** papers · "
            f"**Boolean query:** {ql} · "
            f"_Similarity filter: semantic ≥ {pct}% (keyword Clause rows exempt)._",
            expanded=False,
        ):
            gen = st.session_state.get("search_generation", 0)
            if st.session_state.get("_checkbox_gen_sync") != gen:
                for fp, _ in sorted_papers:
                    st.session_state[f"sel_{fp}"] = True
                st.session_state["_checkbox_gen_sync"] = gen

            for fp, hits in sorted_papers:
                best_score = max(h["score"] for h in hits)
                m0 = hits[0]["metadata"]
                score_pct = int(best_score * 100)
                color = _score_color(best_score)
                pdf_link = pdf_url(fp, PAPERS_DIR)

                has_keyword = any(h.get("match_type") == "keyword" for h in hits)
                has_semantic = any(h.get("match_type") == "semantic" for h in hits)
                if has_keyword and not has_semantic:
                    match_badge = " · 🔑 keyword match"
                elif has_keyword:
                    match_badge = " · 🔑 +keyword"
                else:
                    match_badge = ""

                title_link = (
                    f'<a href="{pdf_link}" target="_blank" '
                    f'style="font-size:1.05em; font-weight:600; '
                    f'text-decoration:none;">'
                    f'📄 {m0["paper_title"]}</a>'
                )
                st.markdown(title_link, unsafe_allow_html=True)
                st.markdown(
                    f':{color}[**{score_pct}% similarity**] · '
                    f'`{Path(fp).name}` · '
                    f'{len(hits)} matching chunk(s){match_badge}',
                    unsafe_allow_html=False,
                )

                abs_rec = _load_abstract_record(fp)
                with st.expander("📋 Abstract", expanded=False):
                    if abs_rec:
                        _render_abstract_record_header(abs_rec)
                        st.caption(
                            f"status: **{abs_rec['status']}** · source: `{abs_rec['source']}` · "
                            f"{abs_rec['char_count']} chars · {abs_rec['word_count']} words"
                        )
                        if abs_rec.get("warnings"):
                            st.caption("⚠️ " + "; ".join(abs_rec["warnings"]))
                        _render_abstract_texts(abs_rec)
                    else:
                        st.warning(
                            "No abstract JSON for this paper. From the app folder run:\n\n"
                            "`python extract_abstracts.py`"
                        )

                with st.expander(f"📑 Matching excerpts ({len(hits)})", expanded=False):
                    keyword_hits = [
                        h for h in hits if h.get("match_type") == "keyword"
                    ]
                    other_hits = [
                        h for h in hits if h.get("match_type") != "keyword"
                    ]

                    if keyword_hits:
                        st.markdown(f"###### Keyword-matching excerpts ({len(keyword_hits)})")
                    for hit in sorted(
                        keyword_hits,
                        key=lambda h: (
                            h.get("keyword_query") or "",
                            float(h.get("lexical_rank") or 0.0),
                            -float(h.get("score") or 0.0),
                        ),
                    ):
                        hm = hit["metadata"]
                        hp = int(hit["score"] * 100)
                        fname_hit = hm.get("file_name") or Path(hm["file_path"]).name
                        page_label = (
                            f"Page {hm['page_num']}"
                            if int(hm.get("page_num") or 0) > 0
                            else "Metadata"
                        )
                        st.markdown(
                            f"**`{fname_hit}`** · **{page_label}** — "
                            f":{_score_color(hit['score'])}[{hp}% similarity] · 🔑 keyword"
                        )
                        _render_hit_text(hit)
                        st.markdown("---")

                    if other_hits:
                        st.markdown(f"###### Semantic / evidence excerpts ({len(other_hits)})")
                    for hit in sorted(other_hits, key=lambda h: h["score"], reverse=True):
                        hm = hit["metadata"]
                        hp = int(hit["score"] * 100)
                        fname_hit = hm.get("file_name") or Path(hm["file_path"]).name
                        page_label = (
                            f"Page {hm['page_num']}"
                            if int(hm.get("page_num") or 0) > 0
                            else "Metadata"
                        )
                        st.markdown(
                            f"**`{fname_hit}`** · **{page_label}** — "
                            f":{_score_color(hit['score'])}[{hp}% similarity]"
                        )
                        _render_hit_text(hit)
                        st.markdown("---")
    elif raw_results:
        st.info(
            f"**Boolean query:** {query_label}\n\n"
            f"No papers above **{int(cutoff * 100)}%** similarity under the current filter "
            f"(semantic chunks only; **Clause** rows set to **keyword** ignore this cutoff)."
        )
    elif query_label:
        st.caption(
            f"Last **Boolean query:** {query_label} — adjust Clauses or cutoff."
        )

    if sorted_papers or manual_only:
        _selection_runner = (
            st.fragment()(_render_paper_selection_widgets)
            if _streamlit_supports_fragment()
            else _render_paper_selection_widgets
        )
        _selection_runner(sorted_papers, manual_only, papers_map, embed_model, gemini_client)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Chat with Gemini (full PDFs)
# ════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("💬 Deep Chat with Gemini")
    st.caption(
        "Gemini reads the **full content** of the selected PDFs. "
        "Upload them first so Gemini can see all papers without size limits."
    )

    deep_papers = st.session_state.get("deep_chat_papers", [])
    deep_history = st.session_state.get("deep_chat_history", [])
    gemini_uploads: dict = st.session_state.get("gemini_uploads", {})

    # ── No papers loaded ──────────────────────────────────────────────────────
    if not deep_papers:
        st.info(
            "No papers loaded yet.\n\n"
            "**How to add papers:**\n"
            "Use the **🔍 Semantic Search** tab → run a search → "
            "select papers at the bottom → click *Send to Deep Chat*"
        )
    else:
        # ── Paper list + upload status ────────────────────────────────────────
        uploaded_count = sum(1 for fp in deep_papers if gemini_uploads.get(fp) is not None)
        failed_count   = sum(1 for fp in deep_papers if fp in gemini_uploads and gemini_uploads[fp] is None)
        pending_count  = len(deep_papers) - uploaded_count - failed_count

        _stage_here = str(st.session_state.get("deep_chat_stage_dir") or "").strip()
        if _stage_here:
            st.caption(f"Latest staged folder (`SELECTED_PDFS_DIR`): `{_stage_here}`")

        with st.expander(
            f"📚 {len(deep_papers)} paper(s) · "
            f"{'✅ All uploaded' if uploaded_count == len(deep_papers) else f'⬆ {pending_count} pending · ✅ {uploaded_count} uploaded' + (f' · ❌ {failed_count} failed' if failed_count else '')}",
            expanded=(uploaded_count < len(deep_papers)),
        ):
            for fp in deep_papers:
                name = Path(fp).name
                link = pdf_url(fp, PAPERS_DIR)
                if gemini_uploads.get(fp) is not None:
                    badge = "✅"
                elif fp in gemini_uploads:
                    badge = "❌"
                else:
                    badge = "⏳"
                st.markdown(
                    f'{badge} <a href="{link}" target="_blank">{name}</a>',
                    unsafe_allow_html=True,
                )

        # ── Upload button ─────────────────────────────────────────────────────
        all_uploaded = uploaded_count == len(deep_papers)

        if not all_uploaded:
            col_up, col_rst = st.columns([3, 1])
            with col_up:
                upload_btn = st.button(
                    f"⬆ Upload {len(deep_papers)} paper(s) to Google Cloud",
                    type="primary",
                    use_container_width=True,
                    help="PDFs are uploaded to Google Cloud Storage and read by Gemini via gs:// URI. No size limits.",
                )
            with col_rst:
                if st.button("🗑 Clear", use_container_width=True, key="clear_deep_pending"):
                    st.session_state["deep_chat_papers"] = []
                    st.session_state["deep_chat_history"] = []
                    st.session_state["gemini_uploads"] = {}
                    st.session_state["deep_chat_stage_dir"] = ""
                    st.rerun()

            if upload_btn:
                status_box = st.empty()
                progress_bar = st.progress(0.0)

                def _on_progress(current, total, name, success, error):
                    """GCS upload progress line for Deep Chat (fraction + per-file status)."""
                    frac = current / total
                    progress_bar.progress(frac)
                    icon = "✅" if success else "❌"
                    msg = f"{icon} ({current}/{total}) {name}"
                    if not success:
                        msg += f" — {error}"
                    status_box.info(msg)

                results = upload_pdfs_to_gcs(
                    pdf_paths=deep_papers,
                    progress_callback=_on_progress,
                )
                st.session_state["gemini_uploads"] = results
                progress_bar.empty()
                status_box.empty()
                st.rerun()
        else:
            col_info, col_rst = st.columns([3, 1])
            with col_info:
                st.success(f"✅ All {len(deep_papers)} paper(s) uploaded — ready to chat.")
            with col_rst:
                if st.button("🗑 Clear", use_container_width=True, key="clear_deep_done"):
                    st.session_state["deep_chat_papers"] = []
                    st.session_state["deep_chat_history"] = []
                    st.session_state["gemini_uploads"] = {}
                    st.session_state["deep_chat_stage_dir"] = ""
                    st.rerun()

        # ── Conversation history & chat ───────────────────────────────────────
        if all_uploaded:
            st.divider()

            # Capture input before rendering history so new messages always
            # appear after existing ones (chat_input floats to bottom of page).
            user_input = st.chat_input(
                "Ask anything about the uploaded papers… "
                "e.g. Summarize the main findings / What methods were used?"
            )

            if deep_history:
                col_hist, col_turns = st.columns([3, 1])
                with col_turns:
                    turns = len([m for m in deep_history if m["role"] == "user"])
                    st.caption(f"{turns} turn(s)")
                if st.button("🗑 Clear conversation", use_container_width=False):
                    st.session_state["deep_chat_history"] = []
                    st.rerun()

            for msg in deep_history:
                role_display = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role_display):
                    st.markdown(msg["content"])

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)

                st.session_state["deep_chat_history"].append(
                    {"role": "user", "content": user_input}
                )

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    prior_history = st.session_state["deep_chat_history"][:-1]

                    try:
                        for chunk in stream_pdf_chat(
                            query=user_input,
                            pdf_paths=deep_papers,
                            chat_history=prior_history,
                            gemini_client=gemini_client,
                            uploaded_files=st.session_state["gemini_uploads"],
                        ):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")

                        placeholder.markdown(full_response)

                    except Exception as e:
                        full_response = f"⚠️ Error: {e}"
                        placeholder.error(full_response)

                st.session_state["deep_chat_history"].append(
                    {"role": "model", "content": full_response}
                )
        else:
            st.info("⬆ Upload the papers to Gemini first, then the chat will appear.")
