# Papers RAG — Installation & Operations Guide (v2.5, public / sanitized)

**Semantic search · FTS keyword · boolean groups · metadata vectors · paste / manual-only papers · Quick Chat · Export · Deep Chat (Vertex AI + GCS) · PubMed → `abstract_meta` JSON**

| Field | Value |
|-------|-------|
| **Updated** | May 2026 |
| **App folder** | `.../papers-rag_app/` (clone or copy wherever you keep the repo) |

This is the **public** edition of the installation guide. Account-specific values are replaced with placeholders:

| Placeholder | Meaning |
|-------------|---------|
| `YOUR_GCP_PROJECT_ID` | Your Google Cloud project id |
| `YOUR_GCS_BUCKET` | Globally unique bucket name you create |
| `YOUR_BILLING_ACCOUNT_ID` | Billing account id (format like `01XXXX-XXXXXX-XXXXXX`) |
| `your-account@example.com` | Your Google login used with `gcloud` |

Section 2 shows **illustrative** command output shapes, not a capture from any one machine.

- **Host environment:** This application has been **tested on Ubuntu** (desktop or server class). Other Linux distributions or macOS may work but are not documented here.
- **Google Cloud SDK:** Install the **`gcloud`** CLI before Vertex / GCS features used by Quick Chat and Deep Chat will work. Follow [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install). On Ubuntu, use Google’s documented **apt** repository or the bundled installer; then run **`gcloud init`** and ADC steps as in [Section 5](#section-5).

## Contents

- [Section 1 — What the app does](#section-1)
- [Section 2 — Example gcloud snapshot](#section-2)
- [Section 3 — Configuration (`.env` and per-library layout)](#section-3)
- [Section 4 — Python environment](#section-4)
- [Section 5 — Install from scratch](#section-5)
- [Section 6 — Operations cheatsheet](#section-6)
- [Section 7 — Billing note](#section-7)
- [Section 8 — v2.5 capabilities snapshot](#section-8)
- [Section 9 — Troubleshooting](#section-9)
- [Section 10 — abstract_meta / PubMed workflow](#section-10)

---

<a id="section-1"></a>
## Section 1: What the app does

This section describes **how the Streamlit app behaves in v2.5** from a user perspective: **Tab 1** — search workbench; **Tab 2** — Deep Chat on full PDFs.

### Configuration and on-disk layout

- **`.env`** (from **`.env.example`**) sets **`PAPERS_DIR`** (your PDF library root, must already exist), **Vertex / GCS** ids (`GCP_PROJECT`, `GCS_BUCKET`, optional `GCP_LOCATION`, `GEMINI_MODEL`), optional **NCBI** credentials, and optional search tuning variables.
- **`papers_rag_config.py`** derives all working folders under **`<PAPERS_DIR>/papers-rag_index/`** — you do **not** configure separate paths for Chroma, `abstract_meta`, exports, or Deep Chat staging in `.env`.
- The sidebar **Papers root** field can switch libraries at runtime; the active path is stored in **`.papers_rag_state.json`** (in the app directory).

```
<PAPERS_DIR>/papers-rag_index/
  chroma_db/                    # Chroma: full-text + metadata vector collections
  keyword_index.sqlite           # SQLite FTS5 keyword index (inside chroma_db/)
  abstract_meta/                 # per-PDF JSON sidecars
  exported_prompts/              # Export context for external LLM
  selected_pdfs/                 # Deep Chat staging (<timestamp>/ per send)
  databases_health_reports/      # TXT + HTML health reports after sync
```

### Sidebar maintenance

Use **🧭 Synchronize/Build/Update Databases** to run the full pipeline in order:

1. Update the **PDF vector** index (chunk, embed, store in Chroma).
2. Rebuild the **keyword FTS** index from chunk text.
3. Extract or refresh missing/stale **`abstract_meta`** JSON (optional PubMed).
4. Rebuild the **metadata vector** index from `abstract_meta` JSON.
5. Write **database health** reports (TXT + HTML).

Individual sidebar controls remain available for each step (PDF vectors, keyword index, `abstract_meta`, metadata vectors) when you prefer partial updates.

Subsections **§1.1–1.8** walk through Tab 1 / Tab 2 behavior in more detail.

### 1.1 Starting point: you do **not** have to run a search first

After the PDF vector index exists, Tab 1 supports:

- **Boolean semantic/keyword search** — optional; multi-clause queries with groups.
- **Paste PDF basenames** — optional; works **without** any search to add **manual-only papers**. Apply resolves names **case-insensitively** against the indexed corpus and **auto-checks** matches (hit-list rows and manual-only rows).

You may use **only** search, **only** paste, or **both**.

<a id="sec-1-2"></a>
### 1.2 Boolean search (Clause rows, operators, groups, diagnostics)

- **Clause rows:** Each row is **semantic** (embedding similarity) or **keyword** (SQLite **FTS5** full-text search on indexed chunks; default backend `fts`).
- **Paper-level logic:** Each clause is evaluated independently and returns a set of matching papers; those paper sets are then combined with **AND**, **OR**, and **NOT**, including **groups** (split after Clause N).
- **Minimum similarity:** Semantic chunks in the **display** list respect the cutoff; **keyword** matches are still shown when they match.
- **Semantic discovery source:** Choose full-text chunks, `.json` metadata vectors, **union**, or **intersection** for semantic clause discovery.
- **Evidence pass:** Runs when you **Search** (not when you chat or export). After boolean discovery, the app fetches additional supporting chunks per final paper (default up to **`EVIDENCE_CHUNKS_PER_PAPER=3`**). These appear in the hit list and in **Export**; they are not re-fetched for Quick Chat.
- **Search diagnostics:** Per-clause timings, cache hits, and discovery stats help tune queries.
- **UI:** Compact boolean query line, translation with semantic/keyword snippets, foldable hit cards (scores, abstracts, multiple excerpts per paper where applicable).

<a id="sec-1-3"></a>
### 1.3 Paste PDF basenames and manual-only papers

- **Apply pasted names** resolves basenames, merges **manual-only papers** into the manual-add list, and sets **`sel_*`** checkboxes for every match.
- Papers already in the **current filtered hit list** appear under *Search hit papers*; **manual-only papers** (pasted names not in that list) appear under *Manual add (not in current search hits)* — both are auto-selected.
- **Clear manual-add list** clears paste state and deselects manual-only papers; papers that remain in the hit list keep their selection.
- **Clear all** resets search, paste, selections, and related Tab 1 state.

### 1.4 Selecting papers — Export, Deep Chat, Quick Chat

- **Select all / Deselect all** apply to search hit papers and **manual-only papers**.
- **Checked rows** drive Export, Send to Deep Chat, and Quick Chat.
- Buttons disable when no rows are checked (caption explains).

### 1.5 Export context (`exported_prompts/prompt_context_<timestamp>.txt`)

- **Checked papers only** — for use with **your preferred LLM** outside the app (paste the `.txt` into ChatGPT, Claude, etc.).
- **Search-hit papers:** `abstract_meta` (when available) plus **all matching excerpts** from the current search in `papers_map` (discovery + evidence chunks).
- **Manual-only papers** (pasted basenames not in the current hit list): **`abstract_meta` only** — no PDF excerpts and no evidence retrieval.

**Why export:** Full retrieval text per hit paper, any external model, and freedom to edit or split the prompt. Best when you need every passage the search surfaced or a chat environment the app does not host.

### 1.6 Quick Chat (Tab 1, Vertex)

In-app streaming chat with **Vertex/Gemini** (requires **`GCP_PROJECT`**, **`GCS_BUCKET`**, and ADC). **Checked papers only** — each message you send includes a context block built from the selection, **not** full PDFs and **not** the whole index.

**What Gemini receives (per checked paper):**

| Paper type | Included in the prompt |
|------------|-------------------------|
| **Search-hit paper** (in the current filtered hit list) | **`abstract_meta` JSON** — typically title, `abstract_text` from the PDF, and `abstract_pubmed` when PubMed enrichment ran — **plus exactly one PDF excerpt**: the **highest-scoring text chunk** from the **last Search** (about 1,000 characters, with filename and page in the source label). That chunk may be from boolean discovery or from the evidence pass, whichever scored highest. |
| **Manual-only paper** (pasted basename, not in the hit list) | **`abstract_meta` JSON only** (same fields as above). **No** PDF excerpts and **no** new retrieval. |

**What Quick Chat does *not* send:**

- The other excerpts listed in the hit-list UI (Export includes those).
- A fresh search or evidence pass when you type a question (context comes from the **last Search** already in session).
- Full PDF files (use **Deep Chat** on Tab 2 for that).

**Why one excerpt per hit paper:** Sending every discovery + evidence chunk for many papers would make prompts very large—slower, costlier, and harder for the model to focus. Quick Chat is for **short, interactive** Q&A; use **Export** when you need **all** excerpts in an external LLM.

Separate chat history from Deep Chat.

<a id="sec-1-7"></a>
### 1.7 Deep Chat (Tab 2) — full PDFs via Gemini + GCS

**Two steps:**

1. **Tab 1 — Send selected papers to Deep Chat** stages **all checked** PDFs under **`selected_pdfs/<YYYYMMDD_HHMMSS>/`** (symlink when possible, copy otherwise). A success banner may list staged filenames.
2. **Tab 2 — Upload to Google Cloud** uploads staged PDFs to **`gs://YOUR_GCS_BUCKET/selected/`**, then enables chat. Gemini reads **`gs://`** URIs so large PDFs are not inline-size limited.

Uses **Application Default Credentials** and **`.env`** project/bucket settings — **not** AI Studio API keys.

### 1.8 PDF links and local server

- Titles and filenames link to **`http://localhost:<PDF_SERVER_PORT>/...`** (default **8502** via **`PDF_SERVER_PORT`** in `.env`; Streamlit default **8501**).

---

<a id="section-2"></a>
## Section 2: Example gcloud snapshot (placeholders — run these on your machine)

Commands were run in a normal shell. **No secrets** (e.g. ADC access tokens) are copied below — only public configuration and API names.

Set these in **`rag_engine.py`** / **`.env`** to match **your** project:

| Field | Value |
|-------|--------|
| Project ID | **YOUR_GCP_PROJECT_ID** (`GCP_PROJECT`) |
| GCS bucket | **YOUR_GCS_BUCKET** (`GCS_BUCKET`) |
| Region | **us-central1** (or your chosen Vertex region; `GCP_LOCATION`) |
| Gemini model | **gemini-2.5-flash** (`GEMINI_MODEL`) |

### 2.1 gcloud version (first lines)

```bash
gcloud --version 2>&1 | head -5
```

Example output (versions change over time):

```text
Google Cloud SDK <version>
alpha <date>
beta <date>
bq <version>
bundled-python3-unix <version>
```

### 2.2 Active configuration

```bash
gcloud config list
```

Example:

```text
[core]
account = your-account@example.com
disable_usage_reporting = True
project = YOUR_GCP_PROJECT_ID

Your active configuration is: [default]
```

### 2.3 Credentialed accounts

```bash
gcloud auth list
```

Example:

```text
Credentialed Accounts
ACTIVE  ACCOUNT
*       your-account@example.com

To set the active account, run:
    $ gcloud config set account `ACCOUNT`
```

### 2.4 Application Default Credentials (ADC)

Check (safe): redirects token to `/dev/null`; prints OK if a token was issued.

```bash
gcloud auth application-default print-access-token >/dev/null 2>&1 && echo "ADC: OK"
```

If ADC works you should see: `ADC: OK`

If this fails, run:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT_ID
```

### 2.5 Enabled APIs (filtered) — project YOUR_GCP_PROJECT_ID

```bash
gcloud services list --enabled --project=YOUR_GCP_PROJECT_ID \
    | grep -E "aiplatform.googleapis|storage.googleapis|generativelanguage"
```

Example lines you want to see:

```text
aiplatform.googleapis.com            Vertex AI API (may show as Agent Platform)
generativelanguage.googleapis.com    Gemini API
storage.googleapis.com               Cloud Storage API
```

(Other enabled services exist; this list is filtered to the ones most relevant.)

<a id="sec-2-6"></a>
### 2.6 Billing linkage for the project

```bash
gcloud billing projects describe YOUR_GCP_PROJECT_ID
```

Example (shape only — your billing account id will differ):

```text
billingAccountName: billingAccounts/YOUR_BILLING_ACCOUNT_ID
billingEnabled: true
name: projects/YOUR_GCP_PROJECT_ID/billingInfo
projectId: YOUR_GCP_PROJECT_ID
```

### 2.7 Buckets in the project

```bash
gcloud storage ls
```

Example:

```text
gs://YOUR_GCS_BUCKET/
```

### 2.8 Objects under `gs://YOUR_GCS_BUCKET/selected/` (snapshot)

```bash
gcloud storage ls gs://YOUR_GCS_BUCKET/selected/
```

If empty, `gcloud` may report no matching URLs — that is normal before uploads. After uploads, re-run:

```bash
gcloud storage ls -l gs://YOUR_GCS_BUCKET/selected/
```

---

<a id="section-3"></a>
## Section 3: Configuration (`.env` and per-library layout)

**Single configuration surface:** Copy **`.env.example`** to **`.env`** in the app folder.

| Category | Variables | Notes |
|----------|-----------|--------|
| **Vertex / GCS** | `GCP_PROJECT`, `GCS_BUCKET` | Required for Quick Chat and Deep Chat. Optional: `GCP_LOCATION`, `GEMINI_MODEL`. |
| **PDF library** | `PAPERS_DIR` | Existing directory containing your PDFs. All indexes and sidecars live under `<PAPERS_DIR>/papers-rag_index/`. |
| **PubMed (optional)** | `NCBI_EMAIL`, `NCBI_API_KEY` | For Entrez during `abstract_meta` extraction. Aliases: `ENTREZ_EMAIL`, `ENTREZ_API_KEY`. |
| **Search tuning (optional)** | `PAPER_DISCOVERY_BATCH_CHUNKS`, `PAPER_DISCOVERY_MAX_CHUNKS`, `EVIDENCE_CHUNKS_PER_PAPER`, `KEYWORD_SEARCH_BACKEND`, `KEYWORD_MAX_PAPERS`, `PDF_SERVER_PORT` | Defaults are in **`.env.example`**. |

**Authentication** for Vertex and GCS uses **Application Default Credentials** (`gcloud auth application-default login`) — complementary to `.env`, not a replacement.

**Library switching:** The sidebar **Papers root** writes **`.papers_rag_state.json`**. On each Streamlit rerun, `app.py` shadows config paths so Chroma, `abstract_meta`, exports, and staging point at the active library without restarting the process.

**Embeddings** (in **`indexer.py`**, not `.env`):

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

**Chroma collections** (under `<PAPERS_DIR>/papers-rag_index/chroma_db/`):

| Collection | Purpose |
|------------|---------|
| `papers` | Full-text PDF chunk vectors |
| `papers_metadata` | Vectors built from `abstract_meta` JSON (title, abstract, authors, etc.) |

**Keyword index:** `keyword_index.sqlite` (FTS5) inside `chroma_db/`; rebuilt from chunk text; can be refreshed without re-embedding PDFs.

**`python-dotenv`** loads `.env` in normal use; **`papers_rag_config`** includes a minimal stdlib fallback if the package is missing (does not override variables already exported in your shell).

---

<a id="section-4"></a>
## Section 4: Python environment (conda env `papers_rag`, snapshot)

Captured with: `conda run -n papers_rag pip freeze | grep -iE "..."`

```text
chromadb==1.5.9
fastembed==0.8.0
google-cloud-storage==3.10.1
google-genai==1.75.0
pandas==3.0.2
PyMuPDF==1.27.2.3
python-dotenv==1.2.2
streamlit==1.57.0
tqdm==4.67.3
```

**Note:** `environment.yml` in the repo should include **google-cloud-storage** in the pip section for fresh installs (add it if `conda env create` fails on GCS upload).

---

<a id="section-5"></a>
## Section 5: Install from scratch (high level)

1. Install Google Cloud SDK: [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. `gcloud init` — log in, select project **YOUR_GCP_PROJECT_ID**.
3. Enable APIs (if not already):

   ```bash
   gcloud services enable aiplatform.googleapis.com storage.googleapis.com \
     --project=YOUR_GCP_PROJECT_ID
   ```

4. ADC + quota project:

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project YOUR_GCP_PROJECT_ID
   ```

5. Create bucket (name must be globally unique):

   ```bash
   gcloud storage buckets create gs://YOUR_GCS_BUCKET \
     --project=YOUR_GCP_PROJECT_ID \
     --location=us-central1 \
     --uniform-bucket-level-access
   ```

6. Conda env: `conda env create -f environment.yml` then `conda activate papers_rag` (add **google-cloud-storage** to pip if missing).
7. `cd` to `papers-rag_app`; copy **`.env.example`** → **`.env`**; set **`PAPERS_DIR`**, **`GCP_PROJECT`**, **`GCS_BUCKET`**, and optional keys.
8. Ensure **`PAPERS_DIR`** exists and contains (or will contain) your PDFs.
9. Run the app:

   ```bash
   streamlit run app.py --server.port=8501
   ```

   → [http://localhost:8501](http://localhost:8501)

10. Sidebar: set **Papers root** if different from `.env`, then run **🧭 Synchronize/Build/Update Databases** once (or **Build PDF Vector Database** first, then keyword + `abstract_meta` + metadata steps). First full index on a large library may take tens of minutes.

---

<a id="section-6"></a>
## Section 6: Operations cheatsheet

**Health checks** (repeat anytime):

```bash
gcloud config list
gcloud auth application-default print-access-token >/dev/null && echo ADC_OK
```

**List uploaded PDFs for Gemini:**

```bash
gcloud storage ls -l gs://YOUR_GCS_BUCKET/selected/
```

**Remove all uploaded PDFs** (matches code path `selected/`):

```bash
gcloud storage rm gs://YOUR_GCS_BUCKET/selected/**
```

**Run the app:**

```bash
conda activate papers_rag
cd /path/to/papers-rag_app
streamlit run app.py --server.port=8501
```

**After adding, removing, or renaming PDFs:** run **Synchronize Databases** (or at minimum update the PDF vector index, then rebuild the keyword index). Check **`databases_health_reports/`** under your library’s `papers-rag_index/` for TXT/HTML sync reports.

**Switch libraries:** change **Papers root** in the sidebar (writes `.papers_rag_state.json`); each library keeps its own `papers-rag_index/` next to its PDFs.

**Refresh `abstract_meta`:** sidebar **📄 Extract / refresh abstract_meta** (typical) or CLI — [Section 10](#section-10).

---

<a id="section-7"></a>
## Section 7: Billing note (Vertex vs AI Studio API key)

This app uses **Vertex AI** via `genai.Client(vertexai=True, ...)` and **ADC**. Charges go to the **Google Cloud billing account** linked to the project (see [Section 2.6](#sec-2-6)).

A **Generative Language / AI Studio API key** uses a separate “prepay” credit pool. If you point the SDK at `api_key=...`, you may see **429 prepayment credits** errors even when GCP credits are fine. The current code path expects ADC.

---

<a id="section-8"></a>
## Section 8: v2.5 capabilities snapshot

**Retrieval stack**

- Chroma **full-text** vectors (`papers`) with incremental indexing and stale-PDF removal.
- SQLite **FTS5** keyword index (`keyword_index.sqlite`) for keyword clauses (default `KEYWORD_SEARCH_BACKEND=fts`).
- Chroma **metadata** vectors (`papers_metadata`) from `abstract_meta` JSON.
- Boolean search: paper-aware semantic discovery, optional metadata/fulltext **union** or **intersection**, post-discovery **evidence** chunks, per-clause **diagnostics**.

**Configuration and layout**

- **`.env`:** `PAPERS_DIR` + cloud ids + optional tuning — not five separate directory roots.
- Per-library bundle: **`<PAPERS_DIR>/papers-rag_index/`**.
- Sidebar **Synchronize Databases** + individual step buttons + health reports.

**Workflows (unchanged in spirit from earlier 2.x)**

- Paste PDF basenames to add **manual-only papers** without search; export / Quick Chat / Deep Chat from checked rows.
- Evidence pass at **Search**; Export = all excerpts for external LLMs; Quick Chat = one best excerpt per hit paper (avoid prompt overload).
- **Manual-only papers:** `abstract_meta` only in Export and Quick Chat.
- Deep Chat: timestamp staging folder → Tab 2 GCS upload → Gemini on full PDFs.
- PubMed enrichment into `abstract_meta` (sidebar or `extract_abstracts.py`).

**Earlier 2.2–2.4 notes (historical):** regex keyword search, repo-adjacent `chroma_db/`, and five-path `.env` layouts were superseded by the current per-library index bundle and FTS/metadata layers. PubMed schema 3 and NCBI error handling (`ncbi_esearch_error`, etc.) remain as documented in [Section 10](#section-10).

---

<a id="section-9"></a>
## Section 9: Troubleshooting (short)

- **ADC errors:** `gcloud auth application-default login` then restart Streamlit.
- **`PAPERS_DIR` / import errors:** ensure **`PAPERS_DIR`** in **`.env`** points at an **existing** folder; restart Streamlit after editing `.env`.
- **Wrong library indexed:** check sidebar **Papers root** and `.papers_rag_state.json`; sync the library you intend.
- **Keyword search empty or stale:** run **Build/Update Keyword Search Index** or full **Synchronize** after PDF vector updates.
- **Metadata semantic clauses empty:** rebuild **Metadata Vector Database** after `abstract_meta` exists; use sidebar sync step 4.
- **No search hits for a gene/tool name:** use a **keyword** clause; lower semantic cutoff only for conceptual queries.
- **Paste “unknown” names / manual-only papers not found:** verify basename spelling and that the PDF is indexed; ambiguous basenames are reported when multiple paths match.
- **Upload failures (Deep Chat):** check `gcloud storage ls gs://YOUR_GCS_BUCKET/selected/` — upload is on Tab 2, not Tab 1 send.
- **Deep Chat 500:** try fewer or smaller PDFs.
- **PDF links broken:** port **8502** in use or change **`PDF_SERVER_PORT`** in `.env`.
- **Large corpus errors:** run **Synchronize** and read health reports under `databases_health_reports/`.
- **PubMed missing or `ncbi_esearch_error`:** see [Section 10](#section-10); set **`NCBI_EMAIL`**; inspect `pubmed_enrichment.status` in JSON.

---

<a id="section-10"></a>
## Section 10: abstract_meta directory: NCBI / PubMed and refresh workflow

Per-PDF JSON lives under **`<PAPERS_DIR>/papers-rag_index/abstract_meta/`** (mirroring relative paths below **`PAPERS_DIR`**). **Two ways to write or refresh files:**

1. **In the Streamlit app (typical):** sidebar section **3. abstract_meta JSON Mirrors** — **📄 Extract / refresh abstract_meta** (same engine as **`extract_abstracts.py`**: incremental vs full refresh, optional PubMed, optional refresh when PDF is newer than JSON).
2. **CLI:** **`extract_abstracts.py`** for batch jobs, logs, and flags — **§10.3**.

The running app **reads** JSON for titles, abstracts, and export; it does **not** call NCBI during ordinary search browsing.

### 10.1 Fields (conceptual)

- **abstract_text** — extracted from the PDF.
- **abstract_pubmed** — Medline-style abstract from **`efetch`** when PubMed enrichment succeeded.
- **pubmed_enrichment** — PMID resolution metadata, **`schema_version`** 3, **`status`** (e.g. **ok**, **no_hit**, **`ncbi_esearch_error`**, …).

### 10.2 NCBI credentials (`.env`)

- **`NCBI_EMAIL`** — contact address for E-utilities (usually your everyday email).
- **`NCBI_API_KEY`** — optional; higher Entrez throughput.
- **`ENTREZ_EMAIL`** / **`ENTREZ_API_KEY`** — accepted aliases.

### 10.3 Refresh or backfill sidescars (in-app and CLI)

**A) Streamlit sidebar** — scope labels match sync incremental mode vs full refresh; toggle PubMed when email is configured.

**B) Command line** — from **`papers-rag_app`** with Conda active:

```bash
conda activate papers_rag
cd /path/to/papers-rag_app
```

**Only PDFs without JSON yet:**

```bash
python extract_abstracts.py --pubmed-meta --only-missing
```

**Full refresh (re-extract all):**

```bash
python extract_abstracts.py --pubmed-meta --force
```

**Only-missing, also refresh when PDF is newer than JSON:**

```bash
python extract_abstracts.py --pubmed-meta --only-missing --refresh-if-newer-pdf
```

**Smoke test:**

```bash
python extract_abstracts.py --pubmed-meta --limit 20
```

**Override paths** (advanced — must match the library you are managing):

```bash
python extract_abstracts.py --papers-dir /path/to/papers/root --pubmed-meta
python extract_abstracts.py --abstract-meta-root /path/to/papers/root/papers-rag_index/abstract_meta --pubmed-meta
```

Run **`python extract_abstracts.py -h`** for full help.

### 10.4 Capturing a full run log

```bash
python extract_abstracts.py --pubmed-meta 2>&1 | tee extract_abstracts_run.log
```

PowerShell:

```powershell
python extract_abstracts.py --pubmed-meta *>&1 | Tee-Object -FilePath extract_abstracts_run.log
```

### 10.5 Repository note

**`papers-rag_index/`** (including `abstract_meta/` and `chroma_db/`) is **gitignored** by default. After **clone**, point **`PAPERS_DIR`** at your PDF folder and run sidebar **Synchronize** or the CLI recipes above.

---

*End — Papers RAG installation & operations guide (v2.5, public / sanitized).*
