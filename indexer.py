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

from papers_paths import PAPERS_DIR, get_all_pdfs

# ── Configuration ────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
DB_PATH = str(APP_DIR / "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # ~130 MB, ONNX-based, fast on CPU
COLLECTION_NAME = "papers"

# Chroma/SQLite may reject one-shot gets over ~999 ids; page metadata reads.
_METADATA_BATCH_SIZE = 400

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


# ── Boolean multi-clause retrieval (paper-level AND / OR / NOT) ────────────────

MAX_BOOLEAN_CLAUSES = 8


def clause_search(
    query: str,
    mode: str,
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
) -> list[dict]:
    """
    Single-clause retrieval: ``semantic`` → embedding only; ``keyword`` → substring only.
    """
    q = (query or "").strip()
    if not q:
        return []
    if mode == "keyword":
        return keyword_search(q, embedding_model, db_path, limit=n_results)
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

    for c in clauses:
        text = (c.get("text") or "").strip()
        mode = c.get("mode", "semantic")
        if mode not in ("semantic", "keyword"):
            mode = "semantic"
        hits = clause_search(text, mode, embedding_model, db_path, n_results)
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

    merged = merge_chunks_for_papers(per_clause_hits, acc)
    return merged, acc, label


def boolean_retrieval(
    clauses: list[dict],
    operators: list[str],
    embedding_model: TextEmbedding,
    db_path: str = DB_PATH,
    n_results: int = 500,
) -> tuple[list[dict], set[str], str]:
    """
    Paper-level left fold with a single segment (no group splits): delegates to
    ``boolean_retrieval_segmented`` with ``split_after=[]``.
    """
    return boolean_retrieval_segmented(
        clauses, [], operators, embedding_model, db_path, n_results
    )


def get_indexed_papers(db_path: str = DB_PATH) -> list[dict]:
    """Return list of all unique indexed papers as {file_path, file_name, paper_title}."""
    seen: dict[str, dict] = {}
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(COLLECTION_NAME)
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
