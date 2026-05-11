# papers-rag_app

**Papers RAG** is a **browser-based** (**Streamlit**) **retrieval-augmented generation** app over your **personal PDF library**: it builds a **local vector index** (**ChromaDB** with **fastembed**) for **semantic and keyword retrieval**, evaluates **multi-clause boolean-style queries**, and **selects** papers with **passage-level excerpts from the chunks that matched the search** (the retrieval evidence behind each hit). Alongside search, each paper can have a companion **`abstract_meta` JSON file** produced offline by **`extract_abstracts.py`**—combining **PDF-derived text and fields** with optional **PubMed / NCBI E-utilities** metadata (`--pubmed-meta`)—so **Quick Chat**, export, and the UI can blend **chunk excerpts from retrieval** with **structured abstract / citation context** when those files exist. The **📄 Export context for external LLM** button writes **`exported_prompts/prompt_context_*.txt`**: checked-paper abstracts plus matching excerpts where the paper appeared in **the current semantic search hits**—aligned with **Quick Chat**. **Manual-only** papers (**Paste PDF basenames** that are **not** in the filtered hit list) contribute **`abstract_meta` JSON–style abstract material only**, with no retrieval excerpts, for export and Quick Chat alike. Paste or upload that **`prompt_context_*.txt`** file into ChatGPT, Claude, or another model when you prefer not to use Vertex.

Conversational workloads use **Google Vertex AI** and **Gemini**, from **matched excerpts** and annotated metadata through **Deep Chat on full PDFs** staged via **Google Cloud Storage** (`gs://`). Development was done with **[Cursor](https://cursor.com)** (including **Claude Sonnet**–assisted coding) and **GitHub MCP** tooling alongside **manual review** of docs and merges.

## Interface preview

**[Open Streamlit UI screenshot (PNG)](./papers_rag_screenshot.png)** — the main search tab has evolved with **multi-clause boolean search**: several **Clause** rows (each **semantic** or **keyword**), **AND / OR / NOT** between rows, an optional **Min similarity** cutoff, and **grouping** so you can parenthesize results (start a new group after a chosen clause). The screenshot may not show every control; the live app is the reference. The same tab provides **Paste PDF basenames** (indexed PDFs resolved case-insensitively; works without running a search), hit selection, **Quick Chat**, **📄 Export context for external LLM** (plaintext under **`exported_prompts/`**; **manual-only** paste rows are abstract-only vs search hits which include excerpts), and related controls.

## How this repository was created

- The application was designed and built using **[Cursor](https://cursor.com)** (AI-assisted coding, refactors, and debugging).
- The **GitHub repository** uses the **GitHub MCP** inside Cursor when helpful (creating or updating content on GitHub programmatically).
- Releases, documentation, installation guides (**`Papers_RAG_*_v*_gcloud_snapshot*.txt`**), README, and merges are reviewed and refined **manually** before pushes.

## Installation and operations

Step-by-step setup, Google Cloud concepts, troubleshooting, a **captured `gcloud` snapshot**, and **`abstract_meta` / PubMed refresh** commands are in:

**[`Papers_RAG_installation_guide_v2.4_gcloud_snapshot.txt`](./Papers_RAG_installation_guide_v2.4_gcloud_snapshot.txt)**

After cloning, copy **`.env.example`** to **`.env`** and set **`GCP_PROJECT`** and **`GCS_BUCKET`** (optional **`GCP_LOCATION`**, **`GEMINI_MODEL`**). For PubMed lookups from **`extract_abstracts.py --pubmed-meta`**, add **NCBI credentials** in **`.env`**: **`NCBI_EMAIL`** is the **contact address** you associate with your NCBI account — often your normal inbox (including Gmail); **`NCBI_API_KEY`** is **optional** and improves Entrez rate limits — see `.env.example` and **Section 10** of the installation guide. Point **`PAPERS_DIR`** in `indexer.py` at your PDF folder. Use **Application Default Credentials** (`gcloud auth application-default login`) — not AI Studio API keys — for Vertex + GCS as described in the guide.

**Refresh `abstract_meta/` (mirrored JSON files per PDF; optional PubMed):** from `papers-rag_app`,  
`conda activate papers_rag` then e.g.

```bash
python extract_abstracts.py --pubmed-meta --only-missing
python extract_abstracts.py --pubmed-meta 2>&1 | tee extract_abstracts_run.log
```

Use **`extract_abstracts.py -h`** for **`--force`**, **`--limit`**, **`--papers-dir`**, **`--refresh-if-newer-pdf`**.

## Run (after environment and GCP are configured)

```bash
conda activate papers_rag
streamlit run app.py
```

- App UI: **http://localhost:8501**
- Local PDF link server: **http://localhost:8502** (see `pdf_server.py`)

## Repository layout

All eight top-level Python modules are **in use** — none are obsolete. **`extract_abstracts.py`** is the **command-line entry point** that walks the corpus via **`papers_paths`** and **`ncbi_pubmed`** and **`abstract_extraction`**. **`app.py`** does **not** import **`extract_abstracts`** or **`ncbi_pubmed`** directly; it uses **`indexer`** (which also uses **`papers_paths`**), **`abstract_extraction`** (reads `abstract_meta` JSON), **`rag_engine`**, and **`pdf_server`**.

| Path | Role |
|------|------|
| `app.py` | Streamlit UI (search clauses, Quick Chat, Deep Chat, selection/export, GCS upload) |
| `indexer.py` | PDF ingest, ChromaDB, hybrid / boolean clause search, keyword regex |
| `rag_engine.py` | Vertex Gemini client, context building (excerpts + abstract-only), GCS, streaming chat |
| `pdf_server.py` | Static HTTP server for clickable local PDF URLs |
| `papers_paths.py` | Shared helpers for corpus paths and PDF discovery |
| `extract_abstracts.py` | **CLI**: batch-write `abstract_meta/*.json` per PDF (`abstract_text`; with `--pubmed-meta`, merges PubMed fields) |
| `abstract_extraction.py` | **Library**: title/abstract/DOI heuristics, JSON schema, **`load`/`save`** for sidescars (used by `extract_abstracts.py` and `app.py`) |
| `ncbi_pubmed.py` | **Library**: NCBI Entrez (`esearch` / `esummary` / `efetch`); enrichment schema **3** — NCBI abstract only at **`abstract_pubmed`**, not duplicated inside enrichment |
| `.env.example` | Template for `.env` (GCP/Vertex + optional NCBI credentials for `--pubmed-meta`) |
| `environment.yml` | Primary Conda environment definition |
| `environment_from_history.yml` | Alternate frozen-ish env export (reference / reproducibility) |
| `requirements.txt` | Pip dependencies |
| `Papers_RAG_installation_guide_v2.4_gcloud_snapshot.txt` | Setup, GCP, troubleshooting, `abstract_meta`/PubMed, `gcloud` snapshot |
| `papers_rag_screenshot.png` | UI screenshot for this README |
| `README.md` | Project overview and quick start |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `selected_pdfs/`, `abstract_meta/`, `*_v2.3_*` install-guide snapshots, extraction logs, vector caches, etc. |

PDFs, extracted abstract metadata folders, and the ChromaDB index are **not** committed (rebuild locally).
