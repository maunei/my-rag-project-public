![Papers RAG — NotebookLM infographic overview](notebooklm_infographic.png)

### 5 minutes NotebookLM Audio Explaining Papers RAG

<a href="https://drive.google.com/file/d/1dN4kUKkY3wGlL1KVHW9-9HSszCJEKyhW/view?usp=sharing" target="_blank" rel="noopener noreferrer"><strong>Listen</strong></a>

## Interface preview

<a href="./papers_rag_screenshot.png" target="_blank" rel="noopener noreferrer"><strong>Open Streamlit UI screenshot (PNG)</strong></a> — Tab 1 shows the clause rows, operators, grouping, paste area, selection, **Quick Chat**, and export; Tab 2 is Deep Chat. The screenshot may lag the live app; run **`streamlit run app.py`** for the current UI.

---

# Papers RAG App

**Papers RAG (v2.5)** is a **Streamlit** app for **retrieval-augmented** work over a **local PDF library** you point to with **`.env`**. It builds and stores a **ChromaDB** vector index (**`fastembed`** embeddings, **`indexer.py`**), indexes PDFs from the sidebar, and serves a **semantic search** and **hybrid keyword** workbench on **Tab 1**, plus **Deep Chat** on **Tab 2** using **Google Vertex AI** (**Gemini**) and **Google Cloud Storage** when you send selected papers for full-document chat.

## Section 1: What the app does (v2.5)

**Corpus and index.** PDFs live under a configured corpus root (**`PAPERS_DIR`**). The app discovers them, chunks text, embeds passages, and persists the index under **`CHROMA_DB_PATH`**. You rebuild or update the index from the sidebar when your library changes.

**Search (Tab 1).** You can run **multi-clause boolean search**: each **Clause** row is either **semantic** (similarity to embeddings) or **keyword** (case-insensitive substring in chunks). Clauses combine at the **paper** level with **AND**, **OR**, and **NOT** (left-to-right within **groups**). You can set a **minimum similarity** cutoff for semantic material; **keyword** hits are still shown when they match. You do **not** need to search first—you can **paste PDF basenames** (one per line); the app resolves them **case-insensitively** against the indexed set and adds them to a **manual-add** list beside any current search hits.

**Evidence and abstracts.** For papers that appear in the **current filtered hit list**, the UI shows **excerpts** from the chunks that matched retrieval. Separately, each paper can have a per-file **`abstract_meta` JSON** mirror under **`ABSTRACT_META_ROOT`**, produced by PDF text heuristics and optionally enriched with **PubMed / NCBI** via **`extract_abstracts.py`** or the sidebar **📄 Extract / refresh abstract_meta** (same logic as the CLI: scope **only missing** vs **full refresh**, optional PubMed). The running app **reads** those JSON files for titles, abstracts, and export—it does not call NCBI during normal browsing.

**Selection, export, and Quick Chat.** Checked rows (from search hits and/or pasted names) drive **📄 Export context for external LLM** (writes **`exported_prompts/prompt_context_*.txt`** under **`EXPORTED_PROMPTS_DIR`**) and **Quick Chat (Vertex)** on Tab 1. For papers that came from **search hits**, export and Quick Chat use **`abstract_meta`** when present **plus** retrieval excerpts. For **manual-only** / paste-only rows (not in the current hit list), export and Quick Chat use **`abstract_meta`-style material only**—no automatic extra chunk retrieval for those paths.

**Deep Chat (Tab 2).** **Send selected papers to Deep Chat** stages **all checked** PDFs under **`SELECTED_PDFS_DIR/<YYYYMMDD_HHMMSS>/`** (symlink when possible, copy otherwise), then uploads to **`gs://`** for **Gemini** so large PDFs are not limited by inline context size. Chat uses **Application Default Credentials** and the project/bucket set in **`.env`**—not AI Studio API keys.

**Local PDF links.** **`pdf_server.py`** serves files at **`http://localhost:<port>/...`** (default **8502** while Streamlit is typically **8501**) so titles and filenames open in the browser.

## How this repository was created

- The application was designed and built using **[Cursor](https://cursor.com)** (AI-assisted coding, refactors, and debugging).
- The **GitHub repository** uses the **GitHub MCP** inside Cursor when helpful (creating or updating content on GitHub programmatically).

## Installation and operations

### Configure `.env` (one checklist: cloud **and** local paths)

Copy [`.env.example`](./.env.example) to **`.env`** and set **all** placeholders that apply to your machine. This app treats configuration as **one place**: you are not done after only “account” or “AI” settings — **`papers_rag_config`** expects **both** (1) **Google Cloud / GCS** identifiers used by Vertex and storage, and (2) **five local directory roots** for your PDF corpus, vector index, `abstract_meta` JSON, export output, and Deep Chat staging. Missing or empty required entries typically surface as **errors at import or when you open the app**, not as gentle warnings.

| What to set | Variables | Purpose |
|-------------|-----------|---------|
| **Vertex / GCS** | `GCP_PROJECT`, `GCS_BUCKET` | Which project and bucket the SDK uses. Optional overrides: `GCP_LOCATION`, `GEMINI_MODEL` (defaults exist — see `.env.example`). |
| **Auth (not pasted into `.env`)** | — | **Application Default Credentials** for Vertex + GCS: `gcloud auth application-default login`. This is separate from filling `.env`, but you need **both** a filled `.env` **and** ADC for chat / Deep Chat. |
| **Local directories (required, v2.5+)** | `PAPERS_DIR` | Root of your PDF library (**must already exist**). |
| | `CHROMA_DB_PATH` | ChromaDB persistence (created if missing). |
| | `ABSTRACT_META_ROOT` | Per-PDF `abstract_meta` JSON mirrors (created if missing). |
| | `EXPORTED_PROMPTS_DIR` | Where **Export context for external LLM** writes `prompt_context_*.txt` (created if missing). |
| | `SELECTED_PDFS_DIR` | Base folder for Deep Chat staging — each run uses a **`YYYYMMDD_HHMMSS`** subfolder. |
| **PubMed (optional)** | `NCBI_EMAIL`, `NCBI_API_KEY` | Only if you use PubMed enrichment (sidebar **📄 Extract / refresh abstract_meta** with PubMed, or **`extract_abstracts.py --pubmed-meta`**). |

### Detailed Installation guide

<a href="./Papers_RAG_installation_guide_v2.5_gcloud_snapshot.md" target="_blank" rel="noopener noreferrer"><strong>Papers-rag V2.5 Installation</strong></a>

**Refresh `abstract_meta/`** (mirrored JSON per PDF): you can do this **from inside the app** with the sidebar **📄 Extract / refresh abstract_meta**—same backfill logic as the CLI (scope: only missing vs full refresh, optional PubMed), so you often **do not need a terminal**. You can also run **`extract_abstracts.py` manually** (batch jobs, scripting, or extra flags). From `papers-rag_app` after `conda activate papers_rag`, examples:

```bash
python extract_abstracts.py --pubmed-meta --only-missing
python extract_abstracts.py --pubmed-meta 2>&1 | tee extract_abstracts_run.log
```

Use **`extract_abstracts.py -h`** for **`--force`**, **`--limit`**, **`--papers-dir`**, **`--abstract-meta-root`**, **`--refresh-if-newer-pdf`**.

## Run (after Conda env, a complete `.env`, and ADC are configured)

```bash
conda activate papers_rag
streamlit run app.py
```

- App UI: **http://localhost:8501**
- Local PDF link server: **http://localhost:8502** (see `pdf_server.py`)

## Repository layout

**Primary Python modules** are documented below — none marked obsolete here. Paths for the corpus, Chroma, `abstract_meta`, exports, and Deep Chat staging are read from **`.env`** via **`papers_rag_config`** (imported indirectly through **`papers_paths`** and **`abstract_extraction`**). **`extract_abstracts.py`** is the **command-line entry point** that walks the corpus via **`papers_paths`** and **`ncbi_pubmed`** and **`abstract_extraction`**. **`app.py`** uses **`extract_abstracts.run_abstract_extractions`** for the sidebar **abstract_meta** button, **`indexer`** (which also uses **`papers_paths`**), **`abstract_extraction`** (reads `abstract_meta` JSON), **`rag_engine`**, and **`pdf_server`**.

| Path | Role |
|------|------|
| `app.py` | Streamlit UI (search clauses, Quick Chat, Deep Chat, selection/export, GCS upload) |
| `indexer.py` | PDF ingest, ChromaDB, hybrid / boolean clause search, keyword regex |
| `rag_engine.py` | Vertex Gemini client, context building (excerpts + abstract-only), GCS, streaming chat |
| `pdf_server.py` | Static HTTP server for clickable local PDF URLs |
| `papers_rag_config.py` | Loads **`.env`**: validates required directory roots (**`PAPERS_DIR`**, **`CHROMA_DB_PATH`**, etc.) |
| `papers_paths.py` | Re-exports corpus root (**`PAPERS_DIR`** from config) + PDF discovery |
| `extract_abstracts.py` | **CLI**: batch-write `abstract_meta/*.json` per PDF (`abstract_text`; with `--pubmed-meta`, merges PubMed fields) |
| `abstract_extraction.py` | **Library**: title/abstract/DOI heuristics, JSON schema, **`load`/`save`** for sidescars (used by `extract_abstracts.py` and `app.py`) |
| `ncbi_pubmed.py` | **Library**: NCBI Entrez (`esearch` / `esummary` / `efetch`); enrichment schema **3** — NCBI abstract only at **`abstract_pubmed`**, not duplicated inside enrichment |
| `.env.example` | Template for `.env` (GCP/Vertex + local paths + optional NCBI credentials) |
| `environment.yml` | Primary Conda environment definition |
| `environment_from_history.yml` | Alternate frozen-ish env export (reference / reproducibility) |
| `requirements.txt` | Pip dependencies |
| `Papers_RAG_installation_guide_v2.5_gcloud_snapshot.md` | Setup / ops guide (GitHub-rendered Markdown; paths in `.env`, sidebar `abstract_meta`, Deep Chat timestamps) |
| `notebooklm_infographic.png` | NotebookLM infographic (README hero) |
| `papers_rag_screenshot.png` | UI screenshot for this README |
| `rag-papers_technical_description.md` | Technical exposition (architecture, retrieval, metadata, modules, Streamlit workflows, interoperability, optional cloud) — [**Papers RAG Technical Description**](#papers-rag-technical-description) |
| `README.md` | Project overview and quick start |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `selected_pdfs/`, `abstract_meta/`, older install-guide snapshot patterns (`*_v2.3_*`, `*_v2.4_*`, …), `*private*`, extraction logs, vector caches, etc. |

PDFs, extracted abstract metadata folders, and the ChromaDB index are **not** committed (rebuild locally).

## Papers RAG Technical Description

The technical architecture and modular design are described in [**`rag-papers_technical_description.md`**](rag-papers_technical_description.md): vector search, hybrid retrieval, metadata pipeline, script roles, Streamlit workflows, export/interoperability, local PDF serving, and optional cloud integration.
