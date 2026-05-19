"""
indexer.py — PDF ingestion and ChromaDB indexing pipeline.

Run this once (or incrementally) to build the vector index from all PDFs.
Already-indexed PDFs are skipped automatically on subsequent runs.
"""

import os
import re
import hashlib
import sqlite3
import json
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from fastembed import TextEmbedding
from tqdm import tqdm

from papers_paths import PAPERS_DIR, get_all_pdfs
from papers_rag_config import ABSTRACT_META_ROOT, CHROMA_DB_PATH

# ── Configuration ────────────────────────────────────────────────────────────

DB_PATH = CHROMA_DB_PATH

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # ~130 MB, ONNX-based, fast on CPU
COLLECTION_NAME = "papers"
METADATA_COLLECTION_NAME = "papers_metadata"

# Chroma/SQLite may reject one-shot gets over ~999 ids; page metadata reads.
_METADATA_BATCH_SIZE = 400
_CHROMA_ID_BATCH_SIZE = 400
_PAPER_DISCOVERY_BATCH_CHUNKS = int(os.getenv("PAPER_DISCOVERY_BATCH_CHUNKS", "5000") or "5000")
_PAPER_DISCOVERY_MAX_CHUNKS = int(os.getenv("PAPER_DISCOVERY_MAX_CHUNKS", "20000") or "20000")
_EVIDENCE_CHUNKS_PER_PAPER = int(os.getenv("EVIDENCE_CHUNKS_PER_PAPER", "3") or "3")

# Sidecar lexical index for proper keyword search. It is derived from Chroma
# chunk text/metadata and can be rebuilt without touching the vector index.
KEYWORD_INDEX_FILENAME = "keyword_index.sqlite"
KEYWORD_INDEX_SCHEMA_VERSION = "1"
KEYWORD_SEARCH_BACKEND = os.getenv("KEYWORD_SEARCH_BACKEND", "fts").strip().lower()
KEYWORD_MAX_PAPERS = int(os.getenv("KEYWORD_MAX_PAPERS", "0") or "0")

CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
EMBED_BATCH_SIZE = 64   # chunks per embedding batch


# ── Chroma helpers ───────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _chroma_client(db_path: str = DB_PATH):
    """Return a cached Chroma PersistentClient for read/query helpers."""
    return chromadb.PersistentClient(path=db_path)


@lru_cache(maxsize=8)
def _chroma_collection(db_path: str = DB_PATH, create: bool = False):
    """Return the papers collection, optionally creating it for indexing."""
    client = _chroma_client(db_path)
    if create:
        return client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=8)
def _metadata_collection(db_path: str = DB_PATH, create: bool = False):
    """Return the paper-level metadata collection, optionally creating it."""
    client = _chroma_client(db_path)
    if create:
        return client.get_or_create_collection(
            METADATA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(METADATA_COLLECTION_NAME)


def _indexed_file_paths_from_collection(collection) -> set[str]:
    """Return all unique indexed file paths from collection metadata."""
    paths: set[str] = set()
    offset = 0
    while True:
        batch = collection.get(
            include=["metadatas"],
            limit=_METADATA_BATCH_SIZE,
            offset=offset,
        )
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        for m in metas:
            if isinstance(m, dict) and m.get("file_path"):
                paths.add(m["file_path"])
        offset += len(ids)
        if len(ids) < _METADATA_BATCH_SIZE:
            break
    return paths


def clear_index_metadata_cache() -> None:
    """Clear cached paper metadata/stat helpers after index changes."""
    get_indexed_papers.cache_clear()


def _delete_indexed_file(collection, file_path: str) -> int:
    """Delete all vector chunks for one indexed PDF path."""
    deleted = 0
    while True:
        batch = collection.get(
            where={"file_path": file_path},
            include=[],
            limit=_CHROMA_ID_BATCH_SIZE,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        collection.delete(ids=ids)
        deleted += len(ids)
        if len(ids) < _CHROMA_ID_BATCH_SIZE:
            break
    return deleted


# ── Keyword FTS helpers ──────────────────────────────────────────────────────

def _keyword_index_path(db_path: str = DB_PATH) -> Path:
    """Return the sidecar SQLite FTS path for a Chroma database directory."""
    return Path(db_path).resolve() / KEYWORD_INDEX_FILENAME


def _keyword_connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open the sidecar keyword SQLite database."""
    path = _keyword_index_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _init_keyword_schema(conn: sqlite3.Connection) -> None:
    """Create keyword sidecar tables if needed."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS keyword_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS keyword_chunks USING fts5(
            chunk_id UNINDEXED,
            text,
            file_path UNINDEXED,
            file_name UNINDEXED,
            paper_title UNINDEXED,
            page_num UNINDEXED,
            chunk_idx UNINDEXED,
            tokenize='unicode61'
        )
        """
    )


def _keyword_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM keyword_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_keyword_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO keyword_meta(key, value) VALUES (?, ?)",
        (key, value),
    )


def _mark_keyword_index_stale(db_path: str = DB_PATH) -> None:
    """Mark sidecar keyword metadata stale after Chroma indexing changes."""
    path = _keyword_index_path(db_path)
    if not path.exists():
        return
    try:
        with _keyword_connect(db_path) as conn:
            _init_keyword_schema(conn)
            _set_keyword_meta(conn, "source_count", "-1")
    except Exception as exc:
        print(f"[WARN] Could not mark keyword index stale: {exc}", flush=True)


def _keyword_index_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM keyword_chunks").fetchone()[0])


def _iter_chroma_documents(collection, batch_size: int = _METADATA_BATCH_SIZE):
    """Yield Chroma chunks as (id, document, metadata) batches."""
    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        yield ids, docs, metas
        offset += len(ids)
        if len(ids) < batch_size:
            break


def _rebuild_keyword_index(
    conn: sqlite3.Connection,
    collection,
    source_count: int,
    progress_callback=None,
) -> None:
    """Rebuild the FTS sidecar from Chroma documents/metadata."""
    conn.execute("DELETE FROM keyword_chunks")
    processed = 0
    for ids, docs, metas in _iter_chroma_documents(collection):
        rows = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            rows.append((
                chunk_id,
                doc or "",
                meta.get("file_path", ""),
                meta.get("file_name", ""),
                meta.get("paper_title", ""),
                str(meta.get("page_num", "")),
                str(meta.get("chunk_idx", "")),
            ))
        conn.executemany(
            """
            INSERT INTO keyword_chunks(
                chunk_id, text, file_path, file_name, paper_title, page_num, chunk_idx
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        processed += len(rows)
        if progress_callback:
            progress_callback(
                min(1.0, processed / max(source_count, 1)),
                f"Keyword chunks {processed:,} of {source_count:,}",
            )
    _set_keyword_meta(conn, "schema_version", KEYWORD_INDEX_SCHEMA_VERSION)
    _set_keyword_meta(conn, "source_count", str(source_count))


def _ensure_keyword_index(collection, db_path: str = DB_PATH) -> None:
    """
    Ensure the sidecar FTS index exists and matches Chroma's chunk count.

    The sidecar is rebuilt from Chroma if missing or stale. This never writes to
    Chroma and never reparses PDFs.
    """
    source_count = int(collection.count())
    with _keyword_connect(db_path) as conn:
        _init_keyword_schema(conn)
        schema_version = _keyword_meta(conn, "schema_version")
        indexed_count = _keyword_meta(conn, "source_count")
        fts_count = _keyword_index_count(conn)
        if (
            schema_version != KEYWORD_INDEX_SCHEMA_VERSION
            or indexed_count != str(source_count)
            or fts_count != source_count
        ):
            _rebuild_keyword_index(conn, collection, source_count)


def rebuild_keyword_index(
    db_path: str = DB_PATH,
    progress_callback=None,
) -> dict:
    """Explicitly rebuild the SQLite FTS keyword index from PDF vector chunks."""
    collection = _chroma_collection(db_path)
    source_count = int(collection.count())
    with _keyword_connect(db_path) as conn:
        _init_keyword_schema(conn)
        _rebuild_keyword_index(
            conn,
            collection,
            source_count,
            progress_callback=progress_callback,
        )
        conn.commit()
    return {"indexed_chunks": source_count}


def get_keyword_index_stats(db_path: str = DB_PATH) -> dict:
    """Return status for the SQLite FTS keyword sidecar."""
    source_count = 0
    try:
        source_count = int(_chroma_collection(db_path).count())
    except Exception:
        pass
    path = _keyword_index_path(db_path)
    if not path.exists():
        return {
            "exists": False,
            "keyword_chunks": 0,
            "source_chunks": source_count,
            "stored_source_chunks": 0,
            "stale": source_count > 0,
        }
    try:
        with _keyword_connect(db_path) as conn:
            _init_keyword_schema(conn)
            keyword_chunks = _keyword_index_count(conn)
            stored_source = int(_keyword_meta(conn, "source_count") or "0")
            schema_version = _keyword_meta(conn, "schema_version")
        stale = (
            schema_version != KEYWORD_INDEX_SCHEMA_VERSION
            or stored_source != source_count
            or keyword_chunks != source_count
        )
        return {
            "exists": True,
            "keyword_chunks": keyword_chunks,
            "source_chunks": source_count,
            "stored_source_chunks": stored_source,
            "stale": stale,
        }
    except Exception:
        return {
            "exists": True,
            "keyword_chunks": 0,
            "source_chunks": source_count,
            "stored_source_chunks": 0,
            "stale": True,
        }


_FTS_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _escape_fts_phrase(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _fts_prefix_terms(part: str) -> list[str]:
    return [m.group(0) for m in _FTS_WORD_RE.finditer(part)]


def _keyword_to_fts_query(query: str) -> str:
    """
    Translate a UI keyword string to a conservative FTS5 MATCH expression.

    Plain multi-word input is treated as a phrase, matching the previous literal
    substring semantics as closely as FTS tokenization allows. Tokens ending in
    ``*`` become prefix matches, e.g. ``neuro*``.
    """
    q = (query or "").strip()
    if not q:
        return ""
    if "*" not in q:
        return _escape_fts_phrase(q)

    terms: list[str] = []
    for raw_part in q.split():
        is_prefix = raw_part.endswith("*")
        part = raw_part[:-1] if is_prefix else raw_part
        words = _fts_prefix_terms(part)
        if not words:
            continue
        if is_prefix:
            if len(words) == 1:
                terms.append(f"{words[0]}*")
            else:
                terms.extend(words[:-1])
                terms.append(f"{words[-1]}*")
        else:
            terms.append(_escape_fts_phrase(" ".join(words)))
    return " AND ".join(terms)


# ── Paper metadata index helpers ─────────────────────────────────────────────

def _as_clean_text(value) -> str:
    """Return a compact string for JSON values used in metadata documents."""
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _pubmed_esummary(record: dict) -> dict:
    enrich = record.get("pubmed_enrichment")
    if isinstance(enrich, dict) and isinstance(enrich.get("esummary"), dict):
        return enrich["esummary"]
    return {}


def _authors_from_record(record: dict) -> str:
    esummary = _pubmed_esummary(record)
    authors = esummary.get("authors")
    if isinstance(authors, list):
        names = [
            _as_clean_text(a.get("name"))
            for a in authors
            if isinstance(a, dict) and _as_clean_text(a.get("name"))
        ]
        if names:
            return "; ".join(names[:40])
    return ""


def _year_from_record(record: dict) -> str:
    esummary = _pubmed_esummary(record)
    for key in ("pubdate", "epubdate", "sortpubdate"):
        text = _as_clean_text(esummary.get(key))
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return m.group(0)
    for key in ("file_name", "title_guess", "title_pdf"):
        text = _as_clean_text(record.get(key))
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return m.group(0)
    return ""


def _doi_from_record(record: dict) -> str:
    doi = _as_clean_text(record.get("doi_for_pubmed") or record.get("doi_canonical_pdf"))
    if doi:
        return doi
    esummary = _pubmed_esummary(record)
    article_ids = esummary.get("articleids")
    if isinstance(article_ids, list):
        for item in article_ids:
            if isinstance(item, dict) and item.get("idtype") == "doi":
                return _as_clean_text(item.get("value"))
    return ""


def _metadata_document_from_record(record: dict) -> str:
    """Build compact paper-level document text from one abstract/PubMed JSON record."""
    esummary = _pubmed_esummary(record)
    title = (
        _as_clean_text(esummary.get("title"))
        or _as_clean_text(record.get("title_pdf"))
        or _as_clean_text(record.get("title_guess"))
    )
    abstract_pubmed = _as_clean_text(record.get("abstract_pubmed"))
    abstract_pdf = _as_clean_text(record.get("abstract_text"))
    abstract = abstract_pubmed or abstract_pdf
    source = _as_clean_text(esummary.get("source"))
    journal = _as_clean_text(esummary.get("fulljournalname")) or source
    pubtypes = esummary.get("pubtype")
    pubtype_text = "; ".join(_as_clean_text(p) for p in pubtypes if _as_clean_text(p)) if isinstance(pubtypes, list) else ""

    fields = [
        ("Title", title),
        ("Authors", _authors_from_record(record)),
        ("Journal", journal),
        ("Year", _year_from_record(record)),
        ("Publication types", pubtype_text),
        ("DOI", _doi_from_record(record)),
        ("PMID", _as_clean_text(esummary.get("uid") or (_pubmed_esummary(record) or {}).get("pmid"))),
        ("Filename", _as_clean_text(record.get("file_name"))),
        ("Abstract", abstract),
    ]
    return "\n".join(f"{label}: {text}" for label, text in fields if text)


def _metadata_record_to_item(record: dict, json_path: Path) -> dict | None:
    """Convert one JSON record to an indexable paper metadata item."""
    file_path = _as_clean_text(record.get("file_path"))
    if not file_path:
        return None
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        return None
    document = _metadata_document_from_record(record)
    if not document.strip():
        return None
    esummary = _pubmed_esummary(record)
    title = (
        _as_clean_text(esummary.get("title"))
        or _as_clean_text(record.get("title_pdf"))
        or _as_clean_text(record.get("title_guess"))
        or _readable_title(file_path)
    )
    authors = _authors_from_record(record)
    metadata = {
        "file_path": str(pdf_path.resolve()),
        "rel_path": _as_clean_text(record.get("rel_path")),
        "file_name": _as_clean_text(record.get("file_name")) or pdf_path.name,
        "paper_title": title,
        "title": title,
        "authors": authors[:2000],
        "year": _year_from_record(record),
        "doi": _doi_from_record(record),
        "pmid": _as_clean_text(esummary.get("uid") or record.get("pmid")),
        "journal": _as_clean_text(esummary.get("fulljournalname") or esummary.get("source")),
        "json_path": str(json_path.resolve()),
        "metadata_source": "abstract_json",
        "page_num": 0,
        "chunk_idx": -1,
    }
    item_id = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()
    return {"id": item_id, "document": document, "metadata": metadata}


def _iter_metadata_index_items(abstract_meta_root: Path = ABSTRACT_META_ROOT):
    """Yield one best JSON-derived metadata item per local PDF file."""
    best_by_file: dict[str, dict] = {}
    for json_path in sorted(Path(abstract_meta_root).rglob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        item = _metadata_record_to_item(record, json_path)
        if not item:
            continue
        fp = item["metadata"]["file_path"]
        current = best_by_file.get(fp)
        if current is None:
            best_by_file[fp] = item
            continue
        # Prefer records with PubMed abstracts and richer document text.
        current_score = len(current["document"]) + (5000 if "Abstract:" in current["document"] else 0)
        item_score = len(item["document"]) + (5000 if "Abstract:" in item["document"] else 0)
        if item_score > current_score:
            best_by_file[fp] = item
    yield from best_by_file.values()


def rebuild_paper_metadata_index(
    db_path: str = DB_PATH,
    embedding_model: TextEmbedding = None,
    abstract_meta_root: Path = ABSTRACT_META_ROOT,
    progress_callback=None,
) -> dict:
    """Rebuild the paper-level metadata Chroma collection from abstract JSON files."""
    model = embedding_model or TextEmbedding(EMBEDDING_MODEL)
    client = _chroma_client(db_path)
    try:
        client.delete_collection(METADATA_COLLECTION_NAME)
        _metadata_collection.cache_clear()
    except Exception:
        pass
    collection = _metadata_collection(db_path, create=True)
    items = list(_iter_metadata_index_items(abstract_meta_root))
    total = len(items)
    stats = {"indexed": 0, "skipped": 0, "errors": 0, "total_records": total}
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = items[start : start + EMBED_BATCH_SIZE]
        if progress_callback:
            progress_callback(
                min(1.0, (start + len(batch)) / max(total, 1)),
                f"Metadata records {start + 1}-{start + len(batch)} of {total}",
            )
        try:
            docs = [item["document"] for item in batch]
            embs = [emb.tolist() for emb in model.embed(docs)]
            collection.add(
                ids=[item["id"] for item in batch],
                documents=docs,
                embeddings=embs,
                metadatas=[item["metadata"] for item in batch],
            )
            stats["indexed"] += len(batch)
        except Exception as exc:
            stats["errors"] += len(batch)
            print(f"[WARN] Metadata index batch failed: {exc}", flush=True)
    return stats


def get_metadata_index_stats(db_path: str = DB_PATH) -> dict:
    """Return paper metadata index stats."""
    try:
        collection = _metadata_collection(db_path)
        return {"total_metadata_papers": int(collection.count())}
    except Exception:
        return {"total_metadata_papers": 0}


def get_abstract_meta_stats(
    abstract_meta_root: Path = ABSTRACT_META_ROOT,
    papers_dir: str = PAPERS_DIR,
) -> dict:
    """Return lightweight JSON mirror stats against the current PDF folder."""
    current_pdfs = {str(Path(p).resolve()) for p in get_all_pdfs(papers_dir)}
    json_total = 0
    linked_existing = 0
    linked_missing = 0
    unreadable = 0
    linked_paths: set[str] = set()
    for json_path in sorted(Path(abstract_meta_root).rglob("*.json")):
        json_total += 1
        try:
            with json_path.open(encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            unreadable += 1
            continue
        if not isinstance(record, dict):
            unreadable += 1
            continue
        fp = _as_clean_text(record.get("file_path"))
        if not fp:
            unreadable += 1
            continue
        resolved_fp = str(Path(fp).resolve())
        if Path(fp).is_file():
            linked_existing += 1
            linked_paths.add(resolved_fp)
        else:
            linked_missing += 1
    missing_json_for_pdfs = len(current_pdfs - linked_paths)
    return {
        "current_pdfs": len(current_pdfs),
        "json_total": json_total,
        "linked_existing": linked_existing,
        "unique_existing_pdfs": len(linked_paths),
        "linked_missing": linked_missing,
        "missing_json_for_pdfs": missing_json_for_pdfs,
        "unreadable": unreadable,
    }


def get_pdf_vector_sync_stats(
    papers_dir: str = PAPERS_DIR,
    db_path: str = DB_PATH,
) -> dict:
    """Compare the current PDF folder with PDF paths stored in the vector index."""
    current_pdfs = {str(Path(p).resolve()) for p in get_all_pdfs(papers_dir)}
    try:
        collection = _chroma_collection(db_path)
        indexed_paths = {str(Path(p).resolve()) for p in _indexed_file_paths_from_collection(collection)}
        total_chunks = int(collection.count())
    except Exception:
        indexed_paths = set()
        total_chunks = 0
    missing_on_disk = indexed_paths - current_pdfs
    unindexed = current_pdfs - indexed_paths
    return {
        "current_pdfs": len(current_pdfs),
        "indexed_papers": len(indexed_paths),
        "total_chunks": total_chunks,
        "missing_on_disk": len(missing_on_disk),
        "unindexed_pdfs": len(unindexed),
    }


def get_database_health_details(
    papers_dir: str = PAPERS_DIR,
    db_path: str = DB_PATH,
    abstract_meta_root: Path = ABSTRACT_META_ROOT,
) -> dict:
    """Return detailed sync diagnostics across PDFs, Chroma, FTS, and JSON metadata."""
    current_pdfs = {str(Path(p).resolve()) for p in get_all_pdfs(papers_dir)}
    try:
        collection = _chroma_collection(db_path)
        indexed_paths = {str(Path(p).resolve()) for p in _indexed_file_paths_from_collection(collection)}
        pdf_vector_chunks = int(collection.count())
    except Exception:
        indexed_paths = set()
        pdf_vector_chunks = 0

    json_records: list[dict] = []
    json_paths_by_pdf: dict[str, list[str]] = {}
    unreadable_json: list[str] = []
    json_missing_pdf: list[dict] = []
    for json_path in sorted(Path(abstract_meta_root).rglob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            unreadable_json.append(str(json_path.resolve()))
            continue
        if not isinstance(record, dict):
            unreadable_json.append(str(json_path.resolve()))
            continue
        fp = _as_clean_text(record.get("file_path"))
        file_name = _as_clean_text(record.get("file_name")) or (Path(fp).name if fp else "")
        title = (
            _as_clean_text(_pubmed_esummary(record).get("title"))
            or _as_clean_text(record.get("title_pdf"))
            or _as_clean_text(record.get("title_guess"))
        )
        row = {
            "json_path": str(json_path.resolve()),
            "file_path": str(Path(fp).resolve()) if fp else "",
            "file_name": file_name,
            "title": title,
        }
        json_records.append(row)
        if not fp:
            json_missing_pdf.append(row)
            continue
        resolved_fp = str(Path(fp).resolve())
        if Path(fp).is_file():
            json_paths_by_pdf.setdefault(resolved_fp, []).append(str(json_path.resolve()))
        else:
            json_missing_pdf.append(row)

    try:
        metadata_collection = _metadata_collection(db_path)
        metadata_vector_records = int(metadata_collection.count())
    except Exception:
        metadata_vector_records = 0

    keyword_stats = get_keyword_index_stats(db_path)
    usable_json_pdfs = set(json_paths_by_pdf)
    linked_existing_json_files = sum(len(paths) for paths in json_paths_by_pdf.values())
    return {
        "pdf_folder": {
            "count": len(current_pdfs),
            "paths": sorted(current_pdfs),
        },
        "pdf_vector_database": {
            "indexed_papers": len(indexed_paths),
            "chunks": pdf_vector_chunks,
            "unindexed_pdfs": sorted(current_pdfs - indexed_paths),
            "indexed_missing_on_disk": sorted(indexed_paths - current_pdfs),
        },
        "keyword_index": keyword_stats,
        "abstract_meta": {
            "json_total": len(json_records) + len(unreadable_json),
            "linked_existing": linked_existing_json_files,
            "unique_existing_pdfs": len(usable_json_pdfs),
            "pdfs_without_json": sorted(current_pdfs - usable_json_pdfs),
            "json_records_missing_pdf": json_missing_pdf,
            "unreadable_json": sorted(set(unreadable_json)),
        },
        "metadata_vector_database": {
            "records": metadata_vector_records,
            "usable_json_records": linked_existing_json_files,
            "unique_existing_pdfs": len(usable_json_pdfs),
            "stale": metadata_vector_records != len(usable_json_pdfs),
        },
    }


# ── Text helpers ──────────────────────────────────────────────────────────────

def _readable_title(filepath: str) -> str:
    """Convert a filename like '130_kania_smartseq2_2018.pdf' → 'Kania Smartseq2 2018'."""
    stem = Path(filepath).stem
    parts = stem.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        stem = parts[1]
    return stem.replace("_", " ").title()


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of ~CHUNK_SIZE characters."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _extract_pages(filepath: str) -> list[dict]:
    """Return list of {page_num, text} dicts from a PDF. Returns [] on failure."""
    pages = []
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_num": page_num, "text": text})
        doc.close()
    except Exception as exc:
        print(f"  [WARN] Could not read {Path(filepath).name}: {exc}")
    return pages


# ── Core indexing ─────────────────────────────────────────────────────────────


def index_papers(
    papers_dir: str = PAPERS_DIR,
    db_path: str = DB_PATH,
    embedding_model: TextEmbedding = None,
    progress_callback=None,
) -> dict:
    """
    Index all PDFs into ChromaDB.

    Args:
        papers_dir:        Root folder containing PDFs.
        db_path:           Where to persist the ChromaDB collection.
        embedding_model:   Pre-loaded SentenceTransformer (loaded here if None).
        progress_callback: Optional fn(fraction: float, message: str) for UI updates.

    Returns:
        Stats dict with keys: indexed, skipped, errors, total_chunks.
    """
    if progress_callback:
        progress_callback(0.0, "Loading embedding model…")

    model = embedding_model or TextEmbedding(EMBEDDING_MODEL)

    pdfs = [str(Path(p).resolve()) for p in get_all_pdfs(papers_dir)]
    total = len(pdfs)
    current_pdf_set = set(pdfs)

    collection = _chroma_collection(db_path, create=True)
    indexed_paths = {str(Path(p).resolve()) for p in _indexed_file_paths_from_collection(collection)}
    stale_paths = sorted(indexed_paths - current_pdf_set)
    stats = {
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "total_chunks": 0,
        "removed_papers": 0,
        "removed_chunks": 0,
    }

    for j, stale_path in enumerate(stale_paths, start=1):
        if progress_callback:
            progress_callback(
                0.0,
                f"Removing stale deleted PDF ({j}/{len(stale_paths)}): {Path(stale_path).name[:70]}",
            )
        removed_chunks = _delete_indexed_file(collection, stale_path)
        if removed_chunks:
            stats["removed_papers"] += 1
            stats["removed_chunks"] += removed_chunks
            indexed_paths.discard(stale_path)

    for i, pdf_path in enumerate(pdfs):
        fraction = (i + 1) / max(total, 1)
        short_name = Path(pdf_path).name[:70]

        if progress_callback:
            progress_callback(fraction, f"({i+1}/{total}) {short_name}")

        # Skip already-indexed files using one metadata scan instead of one Chroma
        # query per PDF during no-op/update runs.
        if pdf_path in indexed_paths:
            stats["skipped"] += 1
            continue

        pages = _extract_pages(pdf_path)
        if not pages:
            stats["errors"] += 1
            continue

        paper_title = _readable_title(pdf_path)
        rel_path = str(Path(pdf_path).relative_to(Path(papers_dir).resolve()))

        all_chunks, all_ids, all_metas = [], [], []
        chunk_global = 0

        for page in pages:
            for chunk in _chunk_text(page["text"]):
                chunk_id = hashlib.md5(
                    f"{pdf_path}::{chunk_global}".encode()
                ).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({
                    "file_path": pdf_path,
                    "rel_path": rel_path,
                    "file_name": Path(pdf_path).name,
                    "paper_title": paper_title,
                    "page_num": page["page_num"],
                    "chunk_idx": chunk_global,
                })
                chunk_global += 1

        if not all_chunks:
            stats["errors"] += 1
            continue

        # Embed and store in batches
        for b in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch_texts = all_chunks[b: b + EMBED_BATCH_SIZE]
            batch_embs = [emb.tolist() for emb in model.embed(batch_texts)]
            collection.add(
                ids=all_ids[b: b + EMBED_BATCH_SIZE],
                documents=batch_texts,
                embeddings=batch_embs,
                metadatas=all_metas[b: b + EMBED_BATCH_SIZE],
            )

        stats["indexed"] += 1
        stats["total_chunks"] += chunk_global
        indexed_paths.add(pdf_path)

    clear_index_metadata_cache()
    if stats["indexed"] > 0 or stats["removed_papers"] > 0:
        _mark_keyword_index_stale(db_path)
    return stats


# ── Query helpers (used by app.py) ────────────────────────────────────────────

def semantic_search(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 6,
    file_paths: list[str] | None = None,
) -> list[dict]:
    """
    Return top-N semantically similar chunks for a query.

    Args:
        query:           Natural language query.
        embedding_model: Loaded TextEmbedding instance.
        db_path:         Path to ChromaDB.
        n_results:       Number of chunks to retrieve.
        file_paths:      If given, restrict search to these PDFs.

    Returns:
        List of dicts with keys: text, metadata, score.
    """
    collection = _chroma_collection(db_path)

    query_emb = [next(iter(embedding_model.embed([query]))).tolist()]

    where = None
    if file_paths:
        where = (
            {"file_path": file_paths[0]}
            if len(file_paths) == 1
            else {"file_path": {"$in": file_paths}}
        )

    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - results["distances"][0][i],  # cosine sim
        })
    return hits


def keyword_search(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    limit: int = 500,
) -> list[dict]:
    """
    Return chunks matching a keyword query using a sidecar SQLite FTS5 index.

    Plain text is matched as a tokenized phrase. A token ending in ``*`` enables
    FTS prefix matching, e.g. ``Sox*`` or ``neuro*``. Chroma remains the source of
    documents, metadata, and embeddings; the sidecar index selects the best
    matching chunk per paper so common terms do not spend the whole result budget
    on repeated chunks from the same PDF. Real cosine-similarity scores are then
    computed against the query embedding so results can still be merged with
    semantic hits on the same scale.

    Set ``KEYWORD_SEARCH_BACKEND=chroma`` to force the previous regex scan path.
    Set ``KEYWORD_MAX_PAPERS`` to a positive integer to cap FTS keyword papers;
    the default ``0`` means no paper cap.
    """
    if KEYWORD_SEARCH_BACKEND == "chroma":
        return _keyword_search_chroma_regex(query, embedding_model, db_path, limit)

    try:
        return _keyword_search_fts(query, embedding_model, db_path, limit)
    except Exception as exc:
        print(f"[WARN] FTS keyword search failed; falling back to Chroma regex: {exc}", flush=True)
        return _keyword_search_chroma_regex(query, embedding_model, db_path, limit)


def _keyword_search_fts(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    limit: int = 500,
) -> list[dict]:
    """Keyword search backed by the derived SQLite FTS sidecar."""
    import numpy as np

    q = (query or "").strip()
    if not q:
        return []

    collection = _chroma_collection(db_path)
    keyword_stats = get_keyword_index_stats(db_path)
    if not keyword_stats.get("exists"):
        raise RuntimeError("SQLite FTS keyword index has not been built")
    if keyword_stats.get("stale"):
        raise RuntimeError("SQLite FTS keyword index is stale")

    match_query = _keyword_to_fts_query(q)
    if not match_query:
        return []

    with _keyword_connect(db_path) as conn:
        _init_keyword_schema(conn)
        rows = conn.execute(
            """
            SELECT chunk_id, file_path, bm25(keyword_chunks) AS lexical_rank
            FROM keyword_chunks
            WHERE keyword_chunks MATCH ?
            ORDER BY lexical_rank
            """,
            (match_query,),
        ).fetchall()

    seen_papers: set[str] = set()
    chunk_ids: list[str] = []
    lexical_ranks: dict[str, float] = {}
    for chunk_id, file_path, lexical_rank in rows:
        paper_key = file_path or chunk_id
        if paper_key in seen_papers:
            continue
        seen_papers.add(paper_key)
        chunk_ids.append(chunk_id)
        lexical_ranks[chunk_id] = float(lexical_rank)
        if KEYWORD_MAX_PAPERS > 0 and len(chunk_ids) >= KEYWORD_MAX_PAPERS:
            break

    if not chunk_ids:
        return []

    query_emb = next(iter(embedding_model.embed([query])))
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)

    by_id: dict[str, dict] = {}
    for b in range(0, len(chunk_ids), _CHROMA_ID_BATCH_SIZE):
        id_batch = chunk_ids[b : b + _CHROMA_ID_BATCH_SIZE]
        results = collection.get(
            ids=id_batch,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        embs = results.get("embeddings")
        if embs is None:
            embs = []
        for chunk_id, doc, meta, emb in zip(ids, docs, metas, embs):
            by_id[chunk_id] = {"document": doc, "metadata": meta, "embedding": emb}

    hits = []
    for chunk_id in chunk_ids:
        row = by_id.get(chunk_id)
        if not row:
            continue
        c_emb = np.array(row["embedding"], dtype=np.float32)
        c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-9)
        score = float(np.dot(q_norm, c_norm))
        hits.append({
            "text": row["document"],
            "metadata": row["metadata"],
            "score": score,
            "match_type": "keyword",
            "keyword_query": q,
            "lexical_rank": lexical_ranks.get(chunk_id),
        })

    return hits


def semantic_search_paper_aware(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
    max_chunks_scanned: int = _PAPER_DISCOVERY_MAX_CHUNKS,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Semantic discovery search that keeps only the best chunk per paper.

    Chroma returns chunks, not papers. For Boolean discovery, repeated chunks
    from the same PDF can crowd out other papers. This helper asks Chroma for
    ranked results from the full vector index, applies ``min_similarity``, then
    deduplicates cutoff-qualified chunks by ``file_path`` so downstream Boolean
    logic works with one representative hit per paper. ``max_chunks_scanned`` is
    the ceiling for cutoff-qualified chunks used during discovery.
    """
    q = (query or "").strip()
    if not q:
        return []

    requested = max(int(n_results), 1)
    scan_limit = max(requested, int(max_chunks_scanned))
    batch_size = max(requested, int(_PAPER_DISCOVERY_BATCH_CHUNKS))
    fetch_n = min(scan_limit, batch_size)
    cutoff = max(float(min_similarity or 0.0), 0.0)

    seen_papers: set[str] = set()
    paper_hits: list[dict] = []
    ranked_chunks_considered = 0
    selected_chunks = 0
    ceiling_reached = False
    cutoff_boundary_reached = False
    fetch_steps: list[int] = []

    while True:
        fetch_steps.append(fetch_n)
        hits = semantic_search(q, embedding_model, db_path, n_results=fetch_n)
        ranked_chunks_considered = len(hits)
        selected_hits = [
            hit for hit in hits if float(hit.get("score", 0.0)) >= cutoff
        ]
        if len(selected_hits) > scan_limit:
            selected_hits = selected_hits[:scan_limit]
        selected_chunks = len(selected_hits)
        cutoff_boundary_reached = selected_chunks < ranked_chunks_considered
        seen_papers.clear()
        paper_hits = []
        for hit in selected_hits:
            hit.setdefault("match_type", "semantic")
            fp = hit.get("metadata", {}).get("file_path")
            if not fp or fp in seen_papers:
                continue
            seen_papers.add(fp)
            paper_hits.append(hit)

        if (
            cutoff_boundary_reached
            or selected_chunks >= scan_limit
            or ranked_chunks_considered < fetch_n
        ):
            ceiling_reached = selected_chunks >= scan_limit
            break
        next_fetch_n = min(scan_limit, fetch_n + batch_size)
        if next_fetch_n == fetch_n:
            ceiling_reached = selected_chunks >= scan_limit
            break
        fetch_n = next_fetch_n

    for hit in paper_hits:
        hit["discovery_chunks_scanned"] = ranked_chunks_considered
        hit["discovery_ranked_chunks_considered"] = ranked_chunks_considered
        hit["discovery_selected_chunks"] = selected_chunks
        hit["discovery_chunk_ceiling_reached"] = ceiling_reached
        hit["discovery_cutoff_boundary_reached"] = cutoff_boundary_reached
        hit["discovery_expanded_to_top_n"] = ranked_chunks_considered
        hit["discovery_batch_size"] = batch_size
        hit["discovery_fetch_steps"] = fetch_steps
        hit["discovery_pagination_mode"] = "expanding_top_n"
        hit["discovery_min_similarity"] = cutoff
        hit["discovery_requested_papers"] = requested
    return paper_hits


def paper_metadata_search(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Semantic paper discovery against the JSON/PubMed metadata collection."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        collection = _metadata_collection(db_path)
    except Exception:
        return []
    query_emb = [next(iter(embedding_model.embed([q]))).tolist()]
    count = int(collection.count())
    if count <= 0:
        return []
    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(max(int(n_results), 1), count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids0 = results.get("ids", [[]])[0]
    for i in range(len(ids0)):
        score = 1.0 - results["distances"][0][i]
        if score < min_similarity:
            continue
        meta = dict(results["metadatas"][0][i] or {})
        meta.setdefault("page_num", 0)
        meta.setdefault("chunk_idx", -1)
        meta.setdefault("match_source", "metadata")
        hits.append({
            "text": results["documents"][0][i],
            "metadata": meta,
            "score": score,
            "match_type": "semantic",
            "discovery_source": "metadata",
            "discovery_selected_chunks": len(hits) + 1,
            "discovery_chunk_ceiling_reached": False,
            "discovery_min_similarity": min_similarity,
        })
    total_selected = len(hits)
    for hit in hits:
        hit["discovery_selected_chunks"] = total_selected
    return hits


def combine_discovery_hits_by_paper(hit_groups: list[list[dict]]) -> list[dict]:
    """Union paper discovery hits, keeping the highest-scoring representative per paper."""
    best: dict[str, dict] = {}
    for hits in hit_groups:
        for hit in hits:
            fp = hit.get("metadata", {}).get("file_path")
            if not fp:
                continue
            if fp not in best or float(hit.get("score", 0.0)) > float(best[fp].get("score", 0.0)):
                best[fp] = hit
    return sorted(best.values(), key=lambda h: float(h.get("score", 0.0)), reverse=True)


def _fulltext_discovery_stats(hits: list[dict]) -> dict:
    """Return aggregate semantic full-text discovery diagnostics."""
    if not hits:
        return {}
    return {
        "discovery_fulltext_representative_chunks": len(hits),
        "discovery_selected_chunks": max(
            int(hit.get("discovery_selected_chunks", len(hits))) for hit in hits
        ),
        "discovery_ranked_chunks_considered": max(
            int(hit.get("discovery_ranked_chunks_considered", len(hits))) for hit in hits
        ),
        "discovery_chunks_scanned": max(
            int(hit.get("discovery_chunks_scanned", len(hits))) for hit in hits
        ),
        "discovery_chunk_ceiling_reached": any(
            bool(hit.get("discovery_chunk_ceiling_reached")) for hit in hits
        ),
        "discovery_cutoff_boundary_reached": any(
            bool(hit.get("discovery_cutoff_boundary_reached")) for hit in hits
        ),
        "discovery_expanded_to_top_n": max(
            int(hit.get("discovery_expanded_to_top_n", len(hits))) for hit in hits
        ),
        "discovery_batch_size": max(
            int(hit.get("discovery_batch_size", 0)) for hit in hits
        ),
        "discovery_fetch_steps": next(
            (
                list(hit.get("discovery_fetch_steps") or [])
                for hit in hits
                if hit.get("discovery_fetch_steps")
            ),
            [],
        ),
    }


def evidence_search_for_papers(
    queries: list[str],
    file_paths: list[str] | set[str],
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    chunks_per_paper: int = _EVIDENCE_CHUNKS_PER_PAPER,
) -> list[dict]:
    """
    Retrieve evidence chunks after Boolean paper discovery.

    Runs semantic search restricted to the selected PDFs for each non-empty query
    and keeps up to ``chunks_per_paper`` chunks per paper across all queries.
    """
    selected_paths = list(dict.fromkeys(fp for fp in file_paths if fp))
    clean_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    if not selected_paths or not clean_queries:
        return []

    per_paper_limit = max(int(chunks_per_paper), 1)
    n_results = max(len(selected_paths) * per_paper_limit, per_paper_limit)
    seen_chunks: set[tuple[str, int]] = set()
    counts_by_paper: dict[str, int] = {}
    evidence: list[dict] = []

    for query in clean_queries:
        hits = semantic_search(
            query,
            embedding_model,
            db_path=db_path,
            n_results=n_results,
            file_paths=selected_paths,
        )
        for hit in hits:
            hit.setdefault("match_type", "semantic")
            meta = hit.get("metadata", {})
            fp = meta.get("file_path")
            if not fp:
                continue
            if counts_by_paper.get(fp, 0) >= per_paper_limit:
                continue
            cid = (fp, int(meta.get("chunk_idx", -1)))
            if cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            counts_by_paper[fp] = counts_by_paper.get(fp, 0) + 1
            h2 = dict(hit)
            h2["evidence_query"] = query
            evidence.append(h2)

    return evidence


def _keyword_search_chroma_regex(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    limit: int = 500,
) -> list[dict]:
    """
    Previous keyword search path: Chroma document regex scan plus cosine scoring.
    Kept as an environment-selectable fallback.
    """
    import numpy as np

    q = (query or "").strip()
    if not q:
        return []

    collection = _chroma_collection(db_path)

    # Case-insensitive substring: treat query as literal text.
    pattern = "(?i)" + re.escape(q)

    results = collection.get(
        where_document={"$regex": pattern},
        include=["documents", "metadatas", "embeddings"],
        limit=limit,
    )

    if not results["ids"]:
        return []

    query_emb = next(iter(embedding_model.embed([query])))
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)

    hits = []
    for i in range(len(results["ids"])):
        c_emb = np.array(results["embeddings"][i], dtype=np.float32)
        c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-9)
        score = float(np.dot(q_norm, c_norm))
        hits.append({
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
            "score": score,
            "match_type": "keyword",
            "keyword_query": q,
        })

    return hits


def hybrid_search(
    query: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
) -> list[dict]:
    """
    Combine semantic search with keyword (exact-substring) search.

    Semantic hits are returned first (tagged match_type='semantic').
    Chunks found only by keyword match are appended (tagged match_type='keyword').
    Keyword-only hits bypass the similarity cutoff in the UI so that tool names,
    gene names, and other proper nouns are always surfaced.

    Keyword matching is case-insensitive (regex with (?i) + escaped literal).
    """
    sem_hits = semantic_search(query, embedding_model, db_path, n_results)
    for h in sem_hits:
        h.setdefault("match_type", "semantic")

    kw_hits: list[dict] = keyword_search(query, embedding_model, db_path)

    if not kw_hits:
        return sem_hits

    seen: set[str] = {
        f"{h['metadata']['file_path']}::{h['metadata']['chunk_idx']}"
        for h in sem_hits
    }

    for hit in kw_hits:
        hit_id = f"{hit['metadata']['file_path']}::{hit['metadata']['chunk_idx']}"
        if hit_id not in seen:
            sem_hits.append(hit)
            seen.add(hit_id)

    return sem_hits


# ── Boolean multi-clause retrieval (paper-level AND / OR / NOT) ────────────────

MAX_BOOLEAN_CLAUSES = 8


def clause_search(
    query: str,
    mode: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
    paper_aware: bool = False,
    min_similarity: float = 0.0,
    semantic_source: str = "fulltext",
) -> list[dict]:
    """
    Single-clause retrieval.

    With ``paper_aware=True``, semantic clauses return one representative chunk
    per paper for Boolean discovery. Keyword clauses are already paper-aware
    through the FTS path.
    """
    q = (query or "").strip()
    if not q:
        return []
    if mode == "keyword":
        return keyword_search(q, embedding_model, db_path, limit=n_results)
    if paper_aware:
        source = (semantic_source or "fulltext").lower()
        fulltext_hits: list[dict] = []
        metadata_hits: list[dict] = []
        if source in ("fulltext", "both", "intersection"):
            fulltext_hits = semantic_search_paper_aware(
                q,
                embedding_model,
                db_path,
                n_results,
                min_similarity=min_similarity,
            )
        if source in ("metadata", "both", "intersection"):
            metadata_hits = paper_metadata_search(
                q,
                embedding_model,
                db_path,
                n_results=n_results,
                min_similarity=min_similarity,
            )
        if source == "metadata":
            return metadata_hits
        if source == "both":
            combined = combine_discovery_hits_by_paper([fulltext_hits, metadata_hits])
            fulltext_papers = len(papers_from_hits(fulltext_hits))
            metadata_papers = len(papers_from_hits(metadata_hits))
            fulltext_stats = _fulltext_discovery_stats(fulltext_hits)
            for hit in combined:
                hit["discovery_source"] = "both"
                hit["discovery_fulltext_papers"] = fulltext_papers
                hit["discovery_metadata_papers"] = metadata_papers
                hit["discovery_combined_papers"] = len(combined)
                hit.update(fulltext_stats)
            return combined
        if source == "intersection":
            fulltext_fps = papers_from_hits(fulltext_hits)
            metadata_fps = papers_from_hits(metadata_hits)
            shared_fps = fulltext_fps & metadata_fps
            combined = combine_discovery_hits_by_paper([
                [hit for hit in fulltext_hits if hit.get("metadata", {}).get("file_path") in shared_fps],
                [hit for hit in metadata_hits if hit.get("metadata", {}).get("file_path") in shared_fps],
            ])
            fulltext_stats = _fulltext_discovery_stats(fulltext_hits)
            for hit in combined:
                hit["discovery_source"] = "intersection"
                hit["discovery_fulltext_papers"] = len(fulltext_fps)
                hit["discovery_metadata_papers"] = len(metadata_fps)
                hit["discovery_combined_papers"] = len(combined)
                hit.update(fulltext_stats)
            return combined
        return fulltext_hits
    hits = semantic_search(q, embedding_model, db_path, n_results)
    for h in hits:
        h.setdefault("match_type", "semantic")
    return hits


def papers_from_hits(hits: list[dict]) -> set[str]:
    """Unique ``file_path`` values appearing in chunk hits."""
    return {h["metadata"]["file_path"] for h in hits}


def merge_chunks_for_papers(
    per_clause_hits: list[list[dict]],
    final_fps: set[str],
) -> list[dict]:
    """Union chunks from all clauses, restricted to ``final_fps``, deduped by paper+chunk_idx."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for hits in per_clause_hits:
        for h in hits:
            fp = h["metadata"]["file_path"]
            if fp not in final_fps:
                continue
            cid = (fp, int(h["metadata"]["chunk_idx"]))
            if cid not in seen:
                seen.add(cid)
                out.append(h)
    return out


def boolean_evidence_queries(clauses: list[dict]) -> list[str]:
    """Return unique clause texts suitable for post-discovery evidence retrieval."""
    queries: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        text = (clause.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(text)
    return queries


def retrieve_boolean_evidence(
    clauses: list[dict],
    final_fps: set[str],
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    chunks_per_paper: int = _EVIDENCE_CHUNKS_PER_PAPER,
) -> list[dict]:
    """Retrieve post-discovery evidence chunks for final Boolean papers."""
    return evidence_search_for_papers(
        boolean_evidence_queries(clauses),
        sorted(final_fps),
        embedding_model,
        db_path=db_path,
        chunks_per_paper=chunks_per_paper,
    )


def neighboring_chunks_for_hit(
    hit: dict,
    db_path: str = DB_PATH,
    window: int = 2,
) -> str:
    """Return nearby chunk text around a hit, for keyword display context."""
    meta = hit.get("metadata") or {}
    fp = meta.get("file_path")
    try:
        chunk_idx = int(meta.get("chunk_idx"))
    except (TypeError, ValueError):
        return hit.get("text") or ""
    if not fp or chunk_idx < 0:
        return hit.get("text") or ""

    lo = max(0, chunk_idx - max(int(window), 0))
    hi = chunk_idx + max(int(window), 0)
    ids = [
        hashlib.md5(f"{fp}::{idx}".encode()).hexdigest()
        for idx in range(lo, hi + 1)
    ]
    try:
        collection = _chroma_collection(db_path)
        results = collection.get(ids=ids, include=["documents", "metadatas"])
    except Exception:
        return hit.get("text") or ""

    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    rows = []
    for doc, row_meta in zip(docs, metas):
        try:
            idx = int((row_meta or {}).get("chunk_idx"))
        except (TypeError, ValueError):
            continue
        rows.append((idx, doc or ""))
    if not rows:
        return hit.get("text") or ""
    rows.sort(key=lambda x: x[0])
    return "\n\n".join(doc for _idx, doc in rows if doc.strip())


def _boolean_clause_slot_labels(n: int) -> list[str]:
    """Display names for boolean rows: ``Clause 1`` … ``Clause n``."""
    return [f"Clause {i + 1}" for i in range(n)]


def _fold_clause_labels(label_bits: list[str], operators: list[str]) -> str:
    """Concatenate clause labels with AND/OR/NOT (same spelling as retrieval)."""
    if not label_bits:
        return ""
    if len(label_bits) == 1:
        return label_bits[0]
    label = label_bits[0]
    for i, op in enumerate(operators):
        op_u = (op or "AND").upper()
        label += f" {op_u} {label_bits[i + 1]}"
    return label


def _fold_paper_sets(
    paper_sets: list[set[str]],
    operators: list[str],
    label_bits: list[str],
) -> tuple[set[str], str]:
    """Left fold over consecutive clause paper sets (within one segment)."""
    if not paper_sets:
        return set(), ""
    if len(paper_sets) == 1:
        return set(paper_sets[0]), label_bits[0]

    acc = set(paper_sets[0])
    for i, op in enumerate(operators):
        nxt = paper_sets[i + 1]
        op_u = (op or "AND").upper()
        if op_u == "AND":
            acc = acc & nxt
        elif op_u == "OR":
            acc = acc | nxt
        elif op_u == "NOT":
            acc = acc - nxt
        else:
            acc = acc & nxt
    label = _fold_clause_labels(label_bits, operators)
    return acc, label


def _markdown_bool_op(op: str) -> str:
    """Streamlit markdown colored span for paper-level boolean operators."""
    op_u = (op or "AND").upper()
    if op_u == "AND":
        return ":green[**AND**]"
    if op_u == "OR":
        return ":orange[**OR**]"
    if op_u == "NOT":
        return ":red[**NOT**]"
    return f":violet[**{op_u}**]"


def _join_detailed_atoms_markdown(atoms: list[str], ops: list[str]) -> str:
    """Join ``(semantic) …`` atoms with colored AND/OR/NOT."""
    if not atoms:
        return ""
    if len(atoms) == 1:
        return atoms[0]
    parts: list[str] = [atoms[0]]
    for i, op in enumerate(ops):
        parts.append(" ")
        parts.append(_markdown_bool_op(op))
        parts.append(" ")
        parts.append(atoms[i + 1])
    return "".join(parts)


def format_boolean_expression_translation_md(
    clauses: list[dict],
    split_after: list[int],
    edge_ops: list[str],
) -> str:
    """
    Markdown suitable for ``st.markdown``: detailed ``(semantic)`` / ``(keyword)`` atoms
    plus **colored** AND / OR / NOT **within** each group and **between** groups
    (truncated text matches retrieval labeling).
    """
    if not clauses:
        return ""
    n = len(clauses)
    if len(edge_ops) != max(0, n - 1):
        return ""

    splits_sorted = sorted({s for s in split_after if 0 <= s <= n - 2})
    detailed_bits: list[str] = []
    for c in clauses:
        text = (c.get("text") or "").strip()
        mode = c.get("mode", "semantic")
        if mode not in ("semantic", "keyword"):
            mode = "semantic"
        short = text[:48] + ("…" if len(text) > 48 else "")
        detailed_bits.append(f"({'keyword' if mode == 'keyword' else 'semantic'}) {short}")

    ranges: list[tuple[int, int]] = []
    start = 0
    for s in splits_sorted:
        ranges.append((start, s))
        start = s + 1
    ranges.append((start, n - 1))

    segments: list[str] = []
    for lo, hi in ranges:
        atoms = detailed_bits[lo : hi + 1]
        ops_inside = [edge_ops[j] for j in range(lo, hi)]
        segments.append(_join_detailed_atoms_markdown(atoms, ops_inside))

    out = segments[0]
    for gi in range(1, len(segments)):
        op_between = _markdown_bool_op(edge_ops[splits_sorted[gi - 1]])
        out += "\n\n" + op_between + "\n\n" + segments[gi]
    return out


def format_boolean_expression_preview(
    clauses: list[dict],
    split_after: list[int],
    edge_ops: list[str],
) -> str:
    """
    Pretty-print the boolean query (groups + operators) **without** hitting the index.
    Uses numbered clauses like ``(Clause 1) OR (Clause 2 AND Clause 3)``, matching
    the label returned by ``boolean_retrieval_segmented``.
    """
    if not clauses:
        return ""
    n = len(clauses)
    if len(edge_ops) != max(0, n - 1):
        return ""

    splits_sorted = sorted({s for s in split_after if 0 <= s <= n - 2})
    label_bits = _boolean_clause_slot_labels(n)

    ranges: list[tuple[int, int]] = []
    start = 0
    for s in splits_sorted:
        ranges.append((start, s))
        start = s + 1
    ranges.append((start, n - 1))

    seg_labels: list[str] = []
    for lo, hi in ranges:
        lbs = label_bits[lo : hi + 1]
        ops_inside = [edge_ops[j] for j in range(lo, hi)]
        lbl_s = _fold_clause_labels(lbs, ops_inside)
        seg_labels.append(f"({lbl_s})")

    label = seg_labels[0]
    for gi in range(1, len(seg_labels)):
        op_u = (edge_ops[splits_sorted[gi - 1]] or "AND").upper()
        label += f" {op_u} {seg_labels[gi]}"
    return label


def boolean_retrieval_segmented(
    clauses: list[dict],
    split_after: list[int],
    edge_ops: list[str],
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
    min_similarity: float = 0.0,
    semantic_source: str = "fulltext",
) -> tuple[list[dict], set[str], str]:
    """
    Paper-level boolean search with optional **groups** (parentheses).

    Clauses are partitioned into contiguous segments. Within each segment,
    ``edge_ops[j]`` for internal edges ``j`` combines consecutive clauses (AND/OR/NOT).
    At a segment boundary after clause index ``s`` (``s`` in ``split_after``),
    ``edge_ops[s]`` combines the segment to the left with the segment to the right.

    Args:
        clauses: Non-empty clause dicts (text + mode).
        split_after: Clause indices ``s`` with ``0 <= s <= n-2``; boundary **after**
            clause ``s`` starts a new group (sorted internally).
        edge_ops: Length ``n - 1``; operator between clause ``j`` and ``j+1``
            (within-group, or between-groups when ``j`` is in ``split_after``).

    Returns:
        (merged_hits, final_paper_paths, human_readable_label) — label uses
        ``Clause 1`` … ``Clause n`` with parentheses per group, e.g.
        ``(Clause 1) OR (Clause 2 AND Clause 3)``.
    """
    if not clauses:
        return [], set(), ""

    n = len(clauses)
    if len(edge_ops) != max(0, n - 1):
        raise ValueError("edge_ops must have length len(clauses) - 1")

    splits_sorted = sorted({s for s in split_after if 0 <= s <= n - 2})
    per_clause_hits: list[list[dict]] = []
    paper_sets: list[set[str]] = []
    label_bits = _boolean_clause_slot_labels(n)
    clause_cache: dict[tuple[str, str], list[dict]] = {}

    for c in clauses:
        text = (c.get("text") or "").strip()
        mode = c.get("mode", "semantic")
        if mode not in ("semantic", "keyword"):
            mode = "semantic"
        cache_key = (mode, text)
        if cache_key not in clause_cache:
            clause_cache[cache_key] = clause_search(
                text,
                mode,
                embedding_model,
                db_path,
                n_results,
                paper_aware=True,
                min_similarity=min_similarity,
                semantic_source=semantic_source,
            )
        hits = list(clause_cache[cache_key])
        per_clause_hits.append(hits)
        paper_sets.append(papers_from_hits(hits))

    ranges: list[tuple[int, int]] = []
    start = 0
    for s in splits_sorted:
        ranges.append((start, s))
        start = s + 1
    ranges.append((start, n - 1))

    seg_sets: list[set[str]] = []
    seg_labels: list[str] = []
    for lo, hi in ranges:
        ps = paper_sets[lo : hi + 1]
        lbs = label_bits[lo : hi + 1]
        ops_inside = [edge_ops[j] for j in range(lo, hi)]
        acc_s, lbl_s = _fold_paper_sets(ps, ops_inside, lbs)
        seg_sets.append(acc_s)
        seg_labels.append(f"({lbl_s})")

    acc = set(seg_sets[0])
    label = seg_labels[0]
    for gi in range(1, len(seg_sets)):
        op_u = (edge_ops[splits_sorted[gi - 1]] or "AND").upper()
        nxt = seg_sets[gi]
        if op_u == "AND":
            acc = acc & nxt
        elif op_u == "OR":
            acc = acc | nxt
        elif op_u == "NOT":
            acc = acc - nxt
        else:
            acc = acc & nxt
        label += f" {op_u} {seg_labels[gi]}"

    discovery_hits = merge_chunks_for_papers(per_clause_hits, acc)
    evidence_hits = retrieve_boolean_evidence(clauses, acc, embedding_model, db_path)
    merged = merge_chunks_for_papers([discovery_hits, evidence_hits], acc)
    if not merged:
        merged = discovery_hits
    return merged, acc, label


def boolean_retrieval(
    clauses: list[dict],
    operators: list[str],
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
    min_similarity: float = 0.0,
    semantic_source: str = "fulltext",
) -> tuple[list[dict], set[str], str]:
    """
    Paper-level left fold with a single segment (no group splits): delegates to
    ``boolean_retrieval_segmented`` with ``split_after=[]``.
    """
    return boolean_retrieval_segmented(
        clauses, [], operators, embedding_model, db_path, n_results, min_similarity, semantic_source
    )


@lru_cache(maxsize=8)
def get_indexed_papers(db_path: str = DB_PATH) -> list[dict]:
    """Return list of all unique indexed papers as {file_path, file_name, paper_title}."""
    seen: dict[str, dict] = {}
    try:
        collection = _chroma_collection(db_path)
        offset = 0
        while True:
            batch = collection.get(
                include=["metadatas"],
                limit=_METADATA_BATCH_SIZE,
                offset=offset,
            )
            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            if not ids:
                break
            for m in metas:
                if not m or not isinstance(m, dict):
                    continue
                fp = m.get("file_path")
                if not fp:
                    continue
                if fp not in seen:
                    seen[fp] = {
                        "file_path": fp,
                        "file_name": m.get("file_name"),
                        "paper_title": m.get("paper_title"),
                        "rel_path": m.get("rel_path"),
                    }
            offset += len(ids)
            if len(ids) < _METADATA_BATCH_SIZE:
                break
        return sorted(seen.values(), key=lambda x: x["paper_title"])
    except Exception as exc:
        print(f"[WARN] get_indexed_papers failed: {exc}", flush=True)
        return []


def is_indexed(db_path: str = DB_PATH) -> bool:
    """Return True if the ChromaDB collection exists and has documents."""
    try:
        collection = _chroma_collection(db_path)
        return collection.count() > 0
    except Exception:
        return False


def get_index_stats(db_path: str = DB_PATH) -> dict:
    """Return {total_chunks, total_papers} for the current index."""
    try:
        collection = _chroma_collection(db_path)
        total_chunks = collection.count()
        papers = get_indexed_papers(db_path)
        return {"total_chunks": total_chunks, "total_papers": len(papers)}
    except Exception:
        return {"total_chunks": 0, "total_papers": 0}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Scanning PDFs in: {PAPERS_DIR}")
    pdfs = get_all_pdfs()
    print(f"Found {len(pdfs)} PDFs")

    def cli_progress(frac, msg):
        """ASCII progress bar for ``python indexer.py`` one-shot indexing."""
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg[:60]:<60}", end="", flush=True)

    stats = index_papers(progress_callback=cli_progress)
    print(f"\n\nDone! Indexed: {stats['indexed']}, Skipped: {stats['skipped']}, "
          f"Errors: {stats['errors']}, Total chunks: {stats['total_chunks']}")
