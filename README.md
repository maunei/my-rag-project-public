# papers-rag_app

Private research assistant: **semantic + keyword hybrid search** over your PDFs (local **ChromaDB** + **fastembed**), plus **Google Gemini** via **Vertex AI** — **Quick Chat** on search excerpts and **Deep Chat** on full PDFs read from **Google Cloud Storage** (`gs://`).

## How this repository was created

- The application was designed and built using **[Cursor](https://cursor.com)** (AI-assisted coding, refactors, and debugging).
- This **GitHub repository** was created from Cursor using the **GitHub MCP** (e.g. `create_repository`, `push_files`).

## Installation and operations

Step-by-step setup, Google Cloud concepts, troubleshooting, and a **captured `gcloud` snapshot** are in:

**[`Papers_RAG_installation_guide_v2.3_gcloud_snapshot.txt`](./Papers_RAG_installation_guide_v2.3_gcloud_snapshot.txt)**

After cloning, point **`PAPERS_DIR`** in `indexer.py` at your PDF folder and set **`GCP_PROJECT`**, **`GCS_BUCKET`**, and region in `rag_engine.py` to match your GCP project. Use **Application Default Credentials** (`gcloud auth application-default login`) — not AI Studio API keys — for Vertex + GCS as described in the guide.

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
| `app.py` | Streamlit UI (search, Quick Chat, Deep Chat, GCS upload) |
| `indexer.py` | PDF ingest, ChromaDB, `hybrid_search` / keyword regex |
| `rag_engine.py` | Vertex Gemini client, GCS upload, streaming chat |
| `pdf_server.py` | Static HTTP server for clickable PDF URLs |
| `environment.yml` / `requirements.txt` | Conda / pip dependencies |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `selected_pdfs/`, etc. |

PDFs and the vector index are **not** committed.
