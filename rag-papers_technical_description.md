# PAPERS-RAG: TECHNICAL SYSTEM EXPOSITION

## Purpose and General Workflow of the Application

Researchers often accumulate hundreds or even thousands of scientific articles in PDF format over many years of work. Eventually, the challenge is no longer obtaining papers, but efficiently identifying which articles are relevant to a biological question, method, or hypothesis—and then working with them in a traceable way.

Papers RAG (v2.5) helps navigate large **local PDF libraries** using semantic retrieval, **FTS keyword** search, **Boolean logic at the paper level**, an independent **`abstract_meta` JSON** metadata layer, and several AI-assisted workflows (in-app Quick Chat, portable export, and full-document Deep Chat on Vertex AI).

The searchable indexes are built **locally** from the user’s PDFs and maintained from the Streamlit sidebar (**Synchronize Databases** or individual step controls). Search results link to originals through a small local PDF HTTP server.

Users may **paste PDF basenames** to add **manual-only papers** (known references not returned by the current search) into the same selection, export, and chat workflows as search hits.

Once papers are selected, the system supports:

- **Quick Chat (Tab 1)** — Vertex/Gemini grounded on `abstract_meta` plus a **single best excerpt** per search-hit paper (focused prompts).
- **Export** — a plaintext bundle with **all** search excerpts for external LLMs (ChatGPT, Claude, etc.).
- **Deep Chat (Tab 2)** — full PDFs staged locally, uploaded to GCS, then conversed with Gemini on complete documents.

The design deliberately avoids locking users into one AI provider: export remains portable; cloud features are optional for indexing and local search.

---

## SYSTEM ARCHITECTURE AND CORE PHILOSOPHY

PAPERS-RAG is a modular system bridging local PDF repositories, structured metadata, and conversational AI (local RAG-style context plus optional Vertex/GCS full-document chat).

Unlike minimal RAG prototypes, the system separates several layers:

- local PDF storage
- **full-text** vector retrieval (Chroma)
- **keyword** retrieval (SQLite FTS5 sidecar)
- **metadata** vector retrieval (Chroma, from `abstract_meta`)
- bibliographic JSON (`abstract_meta`)
- contextual export
- in-app Quick Chat
- optional cloud full-document analysis

### Per-library on-disk bundle

Configuration requires **`PAPERS_DIR`** in **`.env`** (the PDF corpus root must already exist). All derived data for that library lives under:

```
<PAPERS_DIR>/papers-rag_index/
  chroma_db/                  # Chroma persistence (collections + keyword_index.sqlite)
  abstract_meta/              # per-PDF JSON sidecars
  exported_prompts/           # prompt_context_*.txt exports
  selected_pdfs/              # Deep Chat staging (<timestamp>/ per send)
  databases_health_reports/   # sync health TXT + HTML
```

The sidebar **Papers root** can switch libraries at runtime; the active path is stored in **`.papers_rag_state.json`** in the app directory. **`papers_rag_config.py`** resolves paths; **`app.py`** shadows them on each rerun for the active library.

### Parallel infrastructures

1. **Retrieval indexes** — full-text chunks, keyword index, and optional metadata vectors for semantic discovery over titles/abstracts.

2. **Independent metadata** — `abstract_meta/*.json` per PDF, produced locally and optionally enriched via PubMed (Entrez), readable without re-embedding PDFs.

This decoupling allows metadata refresh, keyword rebuilds, and metadata-vector rebuilds without always re-chunking the entire corpus.

---

## 1. FULL-TEXT VECTOR SEARCH (CHROMADB)

The primary semantic engine is a persistent ChromaDB database under **`papers-rag_index/chroma_db/`**, collection **`papers`**.

- **Embedding model:** `BAAI/bge-small-en-v1.5` via **fastembed** (CPU-friendly ONNX).
- **Indexing (`indexer.index_papers`):** PyMuPDF text extraction → overlapping chunks (~1000 characters, 200 overlap) → batch embedding → Chroma storage with metadata (`file_path`, `file_name`, `paper_title`, `page_num`, `chunk_idx`).
- **Incremental behavior:** already-indexed PDFs are skipped via a metadata scan; **stale** entries (indexed paths no longer on disk) are removed on update.
- **Queries:** `semantic_search`, paper-aware discovery (`semantic_search_paper_aware`) for boolean clauses, and scoped `evidence_search_for_papers` after boolean combination.

Chunk-level hits are **aggregated to papers** for boolean logic and UI display.

---

## 2. KEYWORD SEARCH (SQLITE FTS5)

Keyword clauses use a **sidecar** SQLite database: **`chroma_db/keyword_index.sqlite`**, built from Chroma chunk text (default backend **`KEYWORD_SEARCH_BACKEND=fts`**).

- Supports literal and prefix-style token matching (e.g. gene symbols, accession patterns) without relying on Chroma document regex scans.
- Rebuildable **without** re-embedding PDFs (`rebuild_keyword_index`).
- Marked **stale** when the PDF vector index changes; **Synchronize Databases** rebuilds it in step 2.

Keyword hits receive cosine scores for display alongside semantic scores; in the UI, **keyword** matches are not filtered out by the semantic similarity cutoff.

---

## 3. METADATA VECTOR SEARCH (CHROMADB)

A second Chroma collection, **`papers_metadata`**, stores embeddings of text derived from each paper’s **`abstract_meta` JSON** (title, abstracts, authors, etc.).

- Built by **`rebuild_paper_metadata_index`** (sidebar step 4 / sync step 4).
- Powers **semantic discovery source** options: full-text only, metadata only, **union**, or **intersection** when evaluating semantic clauses.

This lets boolean search target bibliographic fields even when the full-text index would miss a match.

---

## 4. HYBRID SEMANTIC + BOOLEAN RETRIEVAL

### Clause evaluation

Each **Clause** row is either:

- **semantic** — embedding similarity (full-text and/or metadata per discovery source), or  
- **keyword** — FTS against **`keyword_index.sqlite`**.

Each clause is evaluated **independently** and yields a **set of papers** (paper-aware semantic discovery deduplicates to one representative chunk per paper for boolean combination).

### Boolean combination

**AND**, **OR**, and **NOT** combine those **paper sets** (not raw chunks), with optional **groups** (split after Clause N). The UI shows a compact boolean label and a human-readable translation.

### Evidence pass (search time only)

After the final paper set is known, **`retrieve_boolean_evidence`** runs additional semantic retrieval restricted to those PDFs, keeping up to **`EVIDENCE_CHUNKS_PER_PAPER`** (default 3) chunks per paper across clause query texts. Results merge into **`search_results`** / **`papers_map`** for the hit list and for **Export**.

The evidence pass does **not** run again when the user opens Quick Chat or Export; it runs when the user clicks **Search**.

### Diagnostics

The UI records per-clause timings, cache behavior, and discovery statistics to help tune queries (`PAPER_DISCOVERY_*` environment limits apply to semantic discovery batching).

---

## 5. INDEPENDENT METADATA PIPELINE (`abstract_meta`)

Each PDF can have a mirrored JSON file under **`papers-rag_index/abstract_meta/`**, produced by **`extract_abstracts.py`** / **`abstract_extraction.py`** and refreshable from the sidebar.

Typical fields include:

- `abstract_text` (from PDF heuristics)
- `abstract_pubmed` (from Entrez when enrichment succeeds)
- `pubmed_enrichment` (schema version 3, status, PMID, errors such as `ncbi_esearch_error`)
- title, DOI candidates, dates, authors, provenance

**`ncbi_pubmed.py`** implements Entrez (`esearch`, `esummary`, `efetch`). The running app **reads** JSON during browse/search/export/chat; NCBI is contacted during extraction when PubMed is enabled, not during ordinary search.

---

## 6. MANUAL-ONLY PAPERS (PASTE PDF BASENAMES)

Users can paste basenames (one per line) without running a search. **`Apply pasted names`** resolves them case-insensitively against the indexed corpus and auto-selects matches.

- Papers already in the **current filtered hit list** appear under *Search hit papers*.
- **Manual-only papers** (resolved but not in the hit list) appear in a separate manual-add list with **`abstract_meta` only** in the UI (no search excerpts).

Manual-only papers follow the same **selection → Export / Quick Chat / Deep Chat** paths as hits, but chunk-based context is limited as described below.

---

## 7. SCRIPT RESPONSIBILITIES AND MODULAR DESIGN

| Module | Role |
|--------|------|
| **`papers_rag_config.py`** | Load `.env`; resolve `PAPERS_DIR`; derive `papers-rag_index/` paths; honor `.papers_rag_state.json` for library switch |
| **`papers_paths.py`** | Corpus root and recursive PDF discovery |
| **`indexer.py`** | PDF ingest; Chroma full-text + metadata collections; FTS keyword index; semantic/keyword/boolean/evidence retrieval; health/sync stats |
| **`extract_abstracts.py`** | CLI batch writer for `abstract_meta` |
| **`abstract_extraction.py`** | Heuristics, JSON schema, load/save helpers |
| **`ncbi_pubmed.py`** | Entrez PubMed enrichment |
| **`rag_engine.py`** | Vertex Gemini client; context assembly; `stream_rag_response` (Quick Chat); GCS upload; Deep Chat streaming |
| **`pdf_server.py`** | Local HTTP PDF links (`PDF_SERVER_PORT`, default 8502) |
| **`app.py`** | Streamlit UI: sync, search, diagnostics, selection, export, Quick Chat, Deep Chat staging/upload |

---

## 8. USER INTERFACE AND WORKFLOW ORGANIZATION

Streamlit: **sidebar** + **Tab 1** (search workbench) + **Tab 2** (Deep Chat).

### Sidebar

- **Papers root** — switch library (`PAPERS_DIR` / state file).
- **Synchronize/Build/Update Databases** — ordered pipeline: PDF vectors → keyword FTS → `abstract_meta` extraction → metadata vectors → health reports.
- Individual controls for each step and **`abstract_meta`** scope (incremental vs full refresh, optional PubMed, PDF-newer-than-JSON).
- Status panels for vector, keyword, metadata, and JSON mirror sync.

### Tab 1: Retrieval, selection, Quick Chat, export

- Multi-clause boolean search, similarity cutoff, semantic discovery source, search diagnostics.
- Paste basenames / **manual-only papers**.
- Hit list: scores, `abstract_meta`, multiple excerpts per paper (discovery + evidence).
- Checked rows drive **Export**, **Send to Deep Chat**, and **Quick Chat**.

### Tab 2: Deep document analysis

1. **Tab 1 — Send selected papers to Deep Chat** stages PDFs under **`selected_pdfs/<YYYYMMDD_HHMMSS>/`** (symlink or copy).
2. **Tab 2 — Upload to Google Cloud** → **`gs://<bucket>/selected/`** → Gemini chat on full documents via **`gs://`** URIs.

---

## 9. QUICK CHAT VS EXPORT (CONTEXT ASSEMBLY)

Both use **`abstract_meta`** for every **checked** paper. They differ in **excerpts** and **destination**.

### Quick Chat (Vertex, Tab 1)

Each user message builds a prompt from **checked papers only**:

| Paper type | Context sent to Gemini |
|------------|-------------------------|
| **Search-hit paper** | `abstract_meta` JSON + **one** PDF excerpt (~1,000 characters): the **highest-scoring chunk** from the **last Search** (`papers_map`; may be discovery or evidence). |
| **Manual-only paper** | `abstract_meta` JSON **only** — no excerpts, no new retrieval. |

**Not included:** other UI excerpts, a fresh search, re-run evidence pass, or full PDFs.

**Rationale:** one excerpt per hit paper keeps in-app prompts bounded for speed, cost, and answer focus. Multi-chunk dumps are left to Export.

Implemented in **`app.py`** (`preloaded_chunks` + `abstracts_by_file_path`) and **`rag_engine.stream_rag_response`**.

### Export (`exported_prompts/prompt_context_<timestamp>.txt`)

| Paper type | Context in file |
|------------|-----------------|
| **Search-hit paper** | `abstract_meta` + **all** matching excerpts from the last search (`papers_map`, including evidence). |
| **Manual-only paper** | `abstract_meta` only |

**Rationale:** portable **full** retrieval context for external LLMs the app does not host; user may edit, split, or archive the file.

Implemented via **`rag_engine.build_external_llm_context_text`**.

### Deep Chat (Tab 2)

Separate path: **entire PDFs** via GCS + Gemini file URIs—not chunk RAG from Tab 1.

---

## 10. CONTEXT EXPORT AND AI INTEROPERABILITY

Export embodies the interoperability principle: retrieval products remain usable outside Vertex.

Users may combine exported abstracts and excerpts with any external chat environment, enabling model comparison, custom prompting, and long-form analysis without streaming through Streamlit.

Quick Chat complements export for **interactive** refinement with a **smaller** grounded context per turn.

---

## 11. LOCAL PDF SERVING AND VERIFICATION

**`pdf_server.py`** serves the active **`PAPERS_DIR`** over HTTP (default port **8502**; Streamlit typically **8501**). Hyperlinks in the UI open PDFs in the browser for verification alongside AI-assisted reading—important when validating claims against source documents.

---

## 12. CLOUD INTEGRATION (OPTIONAL)

**`rag_engine.py`** uses **Vertex AI** (`genai.Client(vertexai=True, ...)`) with **Application Default Credentials** and **`.env`** settings (`GCP_PROJECT`, `GCS_BUCKET`, optional `GCP_LOCATION`, `GEMINI_MODEL`).

Cloud is required for:

- **Quick Chat** (Vertex streaming)
- **Deep Chat** (GCS upload + Gemini on full PDFs)

Local infrastructure—indexing, boolean search, metadata extraction, PDF serving, and **Export**—operates without cloud credentials once indexes exist.

---

## OVERALL OBJECTIVE

PAPERS-RAG turns large local PDF archives into a searchable, metadata-rich research environment:

- **three coordinated indexes** (full-text vectors, FTS keywords, metadata vectors)
- **paper-level boolean** retrieval with an **evidence** pass at search time
- **`abstract_meta`** sidecars with optional PubMed enrichment
- **manual-only** paper inclusion via pasted basenames
- **Quick Chat**, **Export**, and **Deep Chat** tuned to different depth and portability needs

within one modular Streamlit application designed for scientific literature exploration.
