"""
indexer.py — PDF ingestion and ChromaDB indexing pipeline.

Run this once (or incrementally) to build the vector index from all PDFs.
Already-indexed PDFs are skipped automatically on subsequent runs.
"""

import os
import re
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from fastembed import TextEmbedding
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

PAPERS_DIR = "/home/mneira/MAURICIO/papers"
APP_DIR = Path(__file__).parent
DB_PATH = str(APP_DIR / "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # ~130 MB, ONNX-based, fast on CPU
COLLECTION_NAME = "papers"

CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
EMBED_BATCH_SIZE = 64   # chunks per embedding batch

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

def get_all_pdfs(papers_dir: str = PAPERS_DIR) -> list[str]:
    """Recursively collect all PDF paths, sorted alphabetically."""
    return sorted(str(p) for p in Path(papers_dir).rglob("*.pdf"))


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

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = get_all_pdfs(papers_dir)
    total = len(pdfs)
    stats = {"indexed": 0, "skipped": 0, "errors": 0, "total_chunks": 0}

    for i, pdf_path in enumerate(pdfs):
        fraction = (i + 1) / total
        short_name = Path(pdf_path).name[:70]

        if progress_callback:
            progress_callback(fraction, f"({i+1}/{total}) {short_name}")

        # Skip already-indexed files
        existing = collection.get(where={"file_path": pdf_path}, limit=1)
        if existing["ids"]:
            stats["skipped"] += 1
            continue

        pages = _extract_pages(pdf_path)
        if not pages:
            stats["errors"] += 1
            continue

        paper_title = _readable_title(pdf_path)
        rel_path = str(Path(pdf_path).relative_to(papers_dir))

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
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

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
    Return chunks whose stored text contains `query` as a substring, **case-insensitive**.

    Uses ChromaDB's ``where_document`` with ``$regex``: ``(?i)`` + escaped literal pattern,
    so the query is matched as plain text (regex metacharacters in the query are harmless).

    Real cosine-similarity scores are computed against the query embedding so
    results can be merged with semantic hits on the same 0–1 scale.
    """
    import numpy as np

    q = (query or "").strip()
    if not q:
        return []

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    # Case-insensitive substring: treat query as literal text
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


def get_indexed_papers(db_path: str = DB_PATH) -> list[dict]:
    """Return list of all unique indexed papers as {file_path, file_name, paper_title}."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION_NAME)
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        seen: dict[str, dict] = {}
        for m in all_meta:
            fp = m["file_path"]
            if fp not in seen:
                seen[fp] = {
                    "file_path": fp,
                    "file_name": m["file_name"],
                    "paper_title": m["paper_title"],
                    "rel_path": m["rel_path"],
                }
        return sorted(seen.values(), key=lambda x: x["paper_title"])
    except Exception:
        return []


def is_indexed(db_path: str = DB_PATH) -> bool:
    """Return True if the ChromaDB collection exists and has documents."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def get_index_stats(db_path: str = DB_PATH) -> dict:
    """Return {total_chunks, total_papers} for the current index."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION_NAME)
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
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg[:60]:<60}", end="", flush=True)

    stats = index_papers(progress_callback=cli_progress)
    print(f"\n\nDone! Indexed: {stats['indexed']}, Skipped: {stats['skipped']}, "
          f"Errors: {stats['errors']}, Total chunks: {stats['total_chunks']}")
