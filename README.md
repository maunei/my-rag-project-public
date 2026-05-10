# papers-rag_app

Private research assistant: **semantic + keyword hybrid search** over your PDFs (local **ChromaDB** + **fastembed**), plus **Google Gemini** via **Vertex AI** — **Quick Chat** on search excerpts and **Deep Chat** on full PDFs read from **Google Cloud Storage** (`gs://`).

## Interface preview

**[Open Streamlit UI screenshot (PNG)](./papers_rag_screenshot.png)** — main search tab: boolean-style clauses, similarity cutoff, hit selection, Quick Chat, and export.

## How this repository was created

- The application was designed and built using **[Cursor](https://cursor.com)** (AI-assisted coding, refactors, and debugging).
- This **GitHub repository** was created from Cursor using the **GitHub MCP** (e.g. `create_repository`, `push_files`).

## Installation and operations

Step-by-step setup, Google Cloud concepts, troubleshooting, and a **captured `gcloud` snapshot** are in:

**[`Papers_RAG_installation_guide_v2.3_gcloud_snapshot.txt`](./Papers_RAG_installation_guide_v2.3_gcloud_snapshot.txt)**

After cloning, copy **`.env.example`** to **`.env`** and set **`GCP_PROJECT`** and **`GCS_BUCKET`** (optional **`GCP_LOCATION`**, **`GEMINI_MODEL`**). Point **`PAPERS_DIR`** in `indexer.py` at your PDF folder. Use **Application Default Credentials** (`gcloud auth application-default login`) — not AI Studio API keys — for Vertex + GCS as described in the guide.

## Run (after environment and GCP are configured)

```bash
conda activate papers_rag
streamlit run app.py
```

- App UI: **http://localhost:8501**
- Local PDF link server: **http://localhost:8502** (see `pdf_server.py`)

## Repository layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI (search clauses, Quick Chat, Deep Chat, selection/export, GCS upload) |
| `indexer.py` | PDF ingest, ChromaDB, hybrid / boolean clause search, keyword regex |
| `rag_engine.py` | Vertex Gemini client, context building (excerpts + abstract-only), GCS, streaming chat |
| `pdf_server.py` | Static HTTP server for clickable local PDF URLs |
| `papers_paths.py` | Shared path helpers for papers directory layout |
| `extract_abstracts.py` | CLI / workflow to extract abstracts into metadata JSON |
| `abstract_extraction.py` | Abstract extraction logic used by the indexer pipeline |
| `.env.example` | Template for local `.env` (GCP project, bucket, optional model/region) |
| `environment.yml` | Primary Conda environment definition |
| `environment_from_history.yml` | Alternate frozen-ish env export (reference / reproducibility) |
| `requirements.txt` | Pip dependencies |
| `Papers_RAG_installation_guide_v2.3_gcloud_snapshot.txt` | Setup, GCP concepts, troubleshooting, `gcloud` snapshot |
| `papers_rag_screenshot.png` | UI screenshot for this README |
| `README.md` | Project overview and quick start |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `selected_pdfs/`, `abstract_meta/`, vector caches, etc. |

PDFs, extracted abstract metadata folders, and the ChromaDB index are **not** committed (rebuild locally).
