# Papers RAG — Installation & Operations Guide (v2.5, public / sanitized)

**Semantic search · hybrid keyword · boolean groups · paste/manual add · Quick Chat · Deep Chat (Vertex AI + GCS) · PubMed → `abstract_meta` JSON**

| Field | Value |
|-------|-------|
| **Generated** | May 12, 2026 |
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
- **Google Cloud SDK:** Install the **`gcloud`** CLI before Vertex / GCS features used by Deep Chat will work. Follow [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install). On Ubuntu, use Google’s documented **apt** repository or the bundled installer; then run **`gcloud init`** and ADC steps as in [Section 5](#section-5).

## Contents

- [Section 1 — What the app does](#section-1)
- [Section 2 — Example gcloud snapshot](#section-2)
- [Section 3 — Configuration (`.env`)](#section-3)
- [Section 4 — Python environment](#section-4)
- [Section 5 — Install from scratch](#section-5)
- [Section 6 — Operations cheatsheet](#section-6)
- [Section 7 — Billing note](#section-7)
- [Section 8 — Features (v2.2 → v2.5)](#section-8)
- [Section 9 — Troubleshooting](#section-9)
- [Section 10 — abstract_meta / PubMed workflow](#section-10)

---

<a id="section-1"></a>
## Section 1: What the app does

This section describes **how the Streamlit app behaves in v2.5** from a user perspective: **Tab 1** — semantic search and workbench; **Tab 2** — Deep Chat on full PDFs.

**Configuration and layout.** Machine-local roots — **`PAPERS_DIR`**, **`CHROMA_DB_PATH`**, **`ABSTRACT_META_ROOT`**, **`EXPORTED_PROMPTS_DIR`**, **`SELECTED_PDFS_DIR`** — are set in **`.env`** and validated at import by **`papers_rag_config.py`**. The sidebar exposes **Build / Update Index** (Chroma persistence under **`CHROMA_DB_PATH`**; **`fastembed`** embeddings via **`indexer.py`**) and **📄 Extract / refresh abstract_meta**, which runs the same pipeline as **`extract_abstracts.py`**: scope **only missing** vs **full refresh**, optional **PubMed** (requires **`NCBI_EMAIL`** in **`.env`** when enabled), optional **refresh if PDF newer than JSON** when only-missing applies.

**Search and corpus (Tab 1).** PDFs under **`PAPERS_DIR`** are chunked, embedded, and retrieved with **multi-clause boolean search**: each **Clause** is **semantic** or **keyword**; clauses combine at the **paper** level with **AND**, **OR**, and **NOT** (with **groups** and a configurable **minimum similarity** cutoff). You can **paste PDF basenames** (one per line) **without** running a search; names resolve **case-insensitively** against the index. Where a paper appears in the **current filtered hit list**, the UI shows **excerpt** evidence from the chunks that matched.

**`abstract_meta` and PubMed.** Per-PDF JSON mirrors live under **`ABSTRACT_META_ROOT`**. You can generate or refresh them from the sidebar or with **`extract_abstracts.py`** (e.g. **`--pubmed-meta`** for NCBI **Entrez**: `esearch` / `esummary` / `efetch`). The running app **reads** these files for titles, abstracts, and export; NCBI is used during extraction, not during ordinary search browsing. Field names, credentials, CLI recipes, and error **`status`** values such as **`ncbi_esearch_error`** are documented in [Section 10](#section-10).

**Export, Quick Chat, and Deep Chat.** Checked rows drive **📄 Export context for external LLM** (writes **`prompt_context_*.txt`** under **`EXPORTED_PROMPTS_DIR`**), **Quick Chat (Vertex)** on Tab 1, and **Send selected papers to Deep Chat**. Export and Quick Chat pair **`abstract_meta`** with retrieval excerpts for **search-hit** papers; **manual-only** pasted rows are **abstract / JSON only** (aligned behavior). **Deep Chat** stages copies or symlinks under **`SELECTED_PDFS_DIR/<YYYYMMDD_HHMMSS>/`**, uploads to **`gs://`**, and chats with **Gemini** via **Vertex AI** and **ADC** — not AI Studio API keys. **pdf_server.py** serves clickable local PDF URLs (default **8502**; Streamlit **8501**).

Subsections **§1.1–1.8** walk through each area in more detail.

### 1.1 Starting point: you do **not** have to run a search first

After the index is built (sidebar **Build / Update Index**), Tab 1 supports:

- **Boolean semantic search** — optional; retrieves chunk hits per Clause row, then combines **found paper filenames** (indexed PDF paths) with **AND / OR / NOT** at the **paper** level.
- **Paste PDF basenames** — optional; works **without** any search. You paste one `.pdf` filename per line (basename only; paths optional), **Apply**, and the app resolves names **case-insensitively** against the **indexed** corpus (unknown / ambiguous lines are reported).

You may use **only** boolean search, **only** paste, or **both**. Filtered semantic hits can be empty while pasted papers still appear in the manual-add path ([Section 1.3](#sec-1-3)).

### 1.2 Boolean search (Clause rows, operators, groups)

- **Clause rows:** Each row has query text and a mode — **semantic** (embedding similarity) or **keyword** (case-insensitive literal substring in chunks). Up to a configured maximum number of clauses.
- **Hybrid retrieval per clause:** Semantic clauses contribute cosine-ranked chunks; keyword clauses use regex-safe substring matching. **Keyword chunks are always shown** even if they fall below the **Min similarity** cutoff; semantic chunks are filtered by that cutoff when results are displayed.
- **Operators between Clause rows:** **AND** (intersect papers), **OR** (union), **NOT** (set difference on papers), evaluated left-to-right **within** each group.
- **Groups (parentheses):** A multiselect defines *start a new group after Clause N.* Contiguous Clause rows form one group; operators **between** groups combine whole group results. The UI shows:
  - A compact **Boolean query** line using **Clause 1**, **Clause 2**, …
  - A **Translation** line with `(semantic) …` / `(keyword) …` snippets (truncated) and **colored** AND/OR/NOT for readability.
- **After you click Search:** Hit papers appear as foldable cards (scores, abstract expander, excerpt expander). New searches default **all hit checkboxes to selected** (generation counter sync).

<a id="sec-1-3"></a>
### 1.3 Paste PDF basenames — adding papers beside (or without) search hits

- **Apply pasted names** merges resolved paths into an internal **manual-add list** and sets those rows **selected** when added.
- Some pasted names may already appear in the **current filtered hit list** (same PDF as a search hit). Others may be **only** on the manual-add side (“not in current search hits”). The UI separates:
  - **Search hit papers** — scores and **matching excerpts** from retrieval.
  - **Manual add** — checkbox, PDF link, **abstract JSON only** (no search excerpts), tagged as manual where relevant.
- **Clear manual-add list** clears the pasted/manual list and textarea; selection keys for **manual-only** papers are cleared. Papers that **also** appear in the current hit list **keep** their hit-list selection so Export / Deep Chat stay consistent with what you see checked.
- **Clear all** resets search results, manual list, selections, and related UI state more aggressively (see app behavior).

### 1.4 Selecting papers — one workbench for Export, Deep Chat, Quick Chat

- **Select all / Deselect all** apply to **both** hit-list and manual-add paths for the current filters.
- **Checked rows** drive:
  - **Export context for external LLM**
  - **Send selected papers to Deep Chat**
  - **Quick Chat (Vertex)** on Tab 1
- **Export / Deep Chat** disable when **no** checked rows (caption explains).

### 1.5 Export context (`exported_prompts/prompt_context_<timestamp>.txt`)

- Saves **checked papers only**.
- **Papers that came from search hits** for the current retrieval: each gets **abstract_meta-style JSON** (when available) **plus** the **matching chunk texts** that survived filtering — suitable for pasting into ChatGPT, Claude, or another external LLM.
- **Papers that are manual-only / paste-only / search-free** (no hit chunks): the export contains **abstract JSON (+ metadata) only** — **no** automatic extra chunk retrieval for those paths. That keeps behavior aligned with **Quick Chat** ([Section 1.6](#sec-1-6)).
- **Why abstracts still work:** abstracts come from **`abstract_meta`** files produced by **`extract_abstracts.py`** (optionally **`--pubmed-meta`** merges **PubMed** fields into the JSON; see [Section 10](#section-10)), not from semantic hits. If JSON is missing, the UI warns and points you to run extraction.

<a id="sec-1-6"></a>
### 1.6 Quick Chat (Tab 1, Vertex) — excerpt-aware vs abstract-only

- Uses **checked papers only**.
- **Hit papers:** model sees **abstract JSON** plus **best excerpt** material derived from search chunks (filename citations enforced in prompting).
- **Manual-only rows:** **abstract / JSON only** — same rule as export.
- History is separate from Deep Chat; clearing behavior may rerun a fragment or full app depending on Streamlit capabilities.

### 1.7 Deep Chat (Tab 2) — full PDFs via Gemini + GCS

- **Send selected papers to Deep Chat** stages **all checked** PDFs (hits + manual) under **`SELECTED_PDFS_DIR/<YYYYMMDD_HHMMSS>/`** (configured in **`.env`**; default layout often uses a folder named **`selected_pdfs`** off the repo — but the path itself is yours to set). Uses **symlinks** when the OS allows, **copy fallback** when not. Earlier flat **`selected_pdfs/`** wiping behavior does **not** apply: **each send creates a new timestamp subfolder**.
- The app performs a **full rerun** after staging so Tab 2 sees the updated paper list. A **one-shot success banner** (filenames, count cap) may appear on Tab 1 after rerun.
- Tab 2 uploads to **`gs://YOUR_GCS_BUCKET/selected/`** (see `rag_engine` / configuration). Gemini consumes **`gs://`** URIs so large PDFs are not inline-size limited.
- **Chat input** is laid out so the composer stays below the latest exchange.

### 1.8 PDF links and local server

- Paper titles / filenames link to **`http://localhost:<PDF_SERVER_PORT>/...`** served by **`pdf_server.py`** (default port **8502**; Streamlit default is **8501**).

---

<a id="section-2"></a>
## Section 2: Example gcloud snapshot (placeholders — run these on your machine)

Commands were run in a normal shell. **No secrets** (e.g. ADC access tokens) are copied below — only public configuration and API names.

Set these in `rag_engine.py` / `gcloud` to match **your** project:

| Field | Value |
|-------|--------|
| Project ID | **YOUR_GCP_PROJECT_ID** |
| GCS bucket | **YOUR_GCS_BUCKET** |
| Region | **us-central1** (or your chosen Vertex region) |
| Gemini model | **gemini-2.5-flash** |

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
## Section 3: Configuration (`.env` — required local paths + Vertex / PubMed)

**Single configuration surface:** All runtime IDs and paths the app reads from the environment belong in **one file** — **`.env`** (start from **`.env.example`**). There is no separate “AI-only” config: you configure **Google Cloud / Vertex / GCS** targets (**`GCP_PROJECT`**, **`GCS_BUCKET`**, optional **`GCP_LOCATION`**, **`GEMINI_MODEL`**) **together with** the **five mandatory local directory roots** below. Skipping cloud fields breaks chat / GCS features; skipping local roots causes **`papers_rag_config`** validation errors at import. **Authentication** for Vertex and GCS uses **Application Default Credentials** (`gcloud auth application-default login`) — that step complements `.env` but does not replace filling it. Optional **PubMed** credentials use the same **`.env`** file.

**Gemini / GCS:** **`rag_engine.py`** loads **`GCP_PROJECT`**, **`GCS_BUCKET`**, optional **`GCP_LOCATION`**, **`GEMINI_MODEL`** from **`.env`** (see **`.env.example`**).

**v2.5 local roots** (**required**, validated at import by **`papers_rag_config.py`**):

| Variable | Role |
|----------|------|
| **`PAPERS_DIR`** | Existing PDF corpus (directory must exist; not auto-created). |
| **`CHROMA_DB_PATH`** | Folder for **Chromadb PersistentClient** (created if absent). |
| **`ABSTRACT_META_ROOT`** | Mirrored **`*.json`** sidescars (created if absent). |
| **`EXPORTED_PROMPTS_DIR`** | Plaintext **`prompt_context_<timestamp>.txt`** exports. |
| **`SELECTED_PDFS_DIR`** | **Base folder** for Deep Chat staging (**`<timestamp>/`** subdirectories per Tab 1 send). |

**Indexer / embeddings** (still in **`indexer.py`**, **not** in `.env` here):

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

**`papers_paths`** re-exports **`PAPERS_DIR`** from **`papers_rag_config`** only.

`pdf_server.py`:

```python
PDF_SERVER_PORT = 8502   # Streamlit default is 8501
```

**`.env.example`** mirrors all keys above. **`python-dotenv`** is the normal loader; if the package is missing, **`papers_rag_config`** parses **`.env`** with a minimal stdlib fallback (**does not override** variables already exported in your shell).

**NCBI credentials** (PubMed enrichment — **`--pubmed-meta`** or Streamlit checkbox):

```text
NCBI_EMAIL=your.address@gmail.com
  # CONTACT address tied to your NCBI account — typically your ordinary email.

NCBI_API_KEY=…
  # Optional API key from NCBI for higher Entrez throughput.
```

(Aliases `ENTREZ_EMAIL` / `ENTREZ_API_KEY` work the same.)

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

**Note:** `environment.yml` in the repo should include **google-cloud-storage** in the pip section for fresh installs (some copies of the file omit it; add it if `conda env create` fails on GCS upload).

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

5. Create bucket (name must be globally unique — pick your own if taken):

   ```bash
   gcloud storage buckets create gs://YOUR_GCS_BUCKET \
     --project=YOUR_GCP_PROJECT_ID \
     --location=us-central1 \
     --uniform-bucket-level-access
   ```

6. Conda env: `conda env create -f environment.yml` then add **google-cloud-storage** to pip if missing; `conda activate papers_rag`.
7. `cd` to `papers-rag_app`; `streamlit run app.py` → [http://localhost:8501](http://localhost:8501)
8. Sidebar: **Build Index** once; wait for completion.

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
streamlit run app.py
```

---

<a id="section-7"></a>
## Section 7: Billing note (Vertex vs AI Studio API key)

This app uses **Vertex AI** via `genai.Client(vertexai=True, ...)` and **ADC**. Charges go to the **Google Cloud billing account** linked to the project (see [Section 2.6](#sec-2-6)).

A **Generative Language / AI Studio API key** uses a separate “prepay” credit pool. If you point the SDK at `api_key=...`, you may see **429 prepayment credits** errors even when GCP credits are fine. The current code path expects ADC.

---

<a id="section-8"></a>
## Section 8: Features added / evolved (v2.2 → v2.5 snapshot)

**Carried forward from v2.2 (still true):**

- Hybrid search (semantic + case-insensitive regex keyword), merged in indexer.
- Keyword hits bypass the similarity cutoff; UI keyword badges.
- Quick Chat on Tab 1 (excerpts when hits exist; separate session state from Deep Chat).
- Deep Chat composer ordering fix (`chat_input` before history render).
- GCS upload timeouts + `blob.exists()` retry semantics (large PDFs).

**v2.3 additions / refinements (conceptual):**

- **Boolean multi-Clause search** — paper-level AND/OR/NOT; per-Clause semantic vs keyword; optional **groups** (split after Clause N) with live **Boolean query** + **Translation** (semantic/keyword snippets, colored operators).
- **Search-free workflows** — paste basenames, manual-add list, checkbox selection, Export, Quick Chat, and Send to Deep Chat **without** requiring a prior search.
- **Default Min similarity** raised to **0.6** (configurable); keyword Clause rows still exempt from cutoff for display.
- **Paste PDF basenames** — resolve against full index; feedback for matched / unknown / ambiguous; merge with manual-add list; clear-manual behavior preserves selection for papers that remain in the **current hit list**.
- **Export / Quick Chat parity rule** — hits → abstracts + chunks (Quick Chat) or abstracts + matching chunks (export); manual-only / paste-only → **abstract JSON (+ metadata) only** (no automatic extra chunk fetch).
- **Deep Chat staging** — **`SELECTED_PDFS_DIR/<timestamp>/`** (see [Section 3](#section-3) — symlink preferred, copy fallback; optional Tab 2 caption lists latest folder).
- **Indexer robustness** — batched metadata reads for large corpora (SQLite / Chroma variable limits).
- **Sidebar / stats caching** — index stats cached with generation bump after rebuild.
- **UI (v2.3 layout)** — foldable hit/manual sections; fragment-scoped reruns where supported for selection responsiveness.

**v2.4 additions / refinements (conceptual):**

- **PubMed / NCBI in `abstract_meta` JSON** — CLI **`extract_abstracts.py --pubmed-meta`** calls **`ncbi_pubmed.py`**: **`esearch`** (DOI `[doi]` first, **title `[Title]`** fallback), **`esummary`**, **`efetch`** abstract XML; stores **`pubmed_enrichment`** (schema 3) plus top-level **`abstract_pubmed`** when available.
- **Entrez `esearch` ERROR handling** — when NCBI returns **`esearchresult.ERROR`** (often **no `idlist`**), responses map to **`ncbi_esearch_error`** instead of a misleading **`parse_error`**; **one automatic retry** on first NCBI-error payload; DOI-route failures with that status still **trigger title fallback**.
- **`.env` NCBI credentials** — **`NCBI_EMAIL`** is the **contact address** you pair with Entrez (**often your everyday email**, e.g. Gmail); **`NCBI_API_KEY`** is optional for throughput. **`ENTREZ_EMAIL`** / **`ENTREZ_API_KEY`** synonyms are honored.

**v2.5 additions / refinements (conceptual):**

- **`.env` machine roots** (**`papers_rag_config`**): **`PAPERS_DIR`**, **`CHROMA_DB_PATH`**, **`ABSTRACT_META_ROOT`**, **`EXPORTED_PROMPTS_DIR`**, **`SELECTED_PDFS_DIR`** (see [Section 3](#section-3)).
- **Sidebar `abstract_meta`** — **📄 Extract / refresh** in the UI mirrors **`extract_abstracts`** (scope **only missing** vs **full refresh**, PubMed checkbox, optional **PDF newer than JSON**), so routine sidescar updates need not use a terminal. **Scope `on_change`** clears the checkbox when switching to **full refresh** (**must not live inside `st.form`** — Streamlit only allows **`st.form_submit_button`** callbacks inside forms).
- **Deep Chat staging timestamps** — per-send folder under **`SELECTED_PDFS_DIR`**.
- **Chromadb directory** comes from **`CHROMA_DB_PATH`** (was previously implied under **`chroma_db/`** beside the repo).
- **UI** — **Papers RAG V2.5** page title / sidebar heading.

---

<a id="section-9"></a>
## Section 9: Troubleshooting (short)

- **ADC errors:** `gcloud auth application-default login` then restart Streamlit.
- **No search hits for a tool name:** rely on keyword Clause rows; lower semantic cutoff for conceptual queries only.
- **Paste “unknown” names:** verify basename spelling and index membership; ambiguous basenames are skipped when multiple indexed paths match.
- **Upload failures:** check `gcloud storage ls` — file may exist despite UI error.
- **Deep Chat 500:** try fewer/lighter PDFs or exclude a huge file in the prompt.
- **PDF links broken:** port 8502 in use — check `pdf_server.PDF_SERVER_PORT`.
- **Large corpus listing errors:** ensure indexer batching is present; rebuild env.
- **Missing-directory `RuntimeError` at import (`PAPERS_DIR`, `CHROMA_DB_PATH`, …):** fill every required root in **`.env`** (**`.env.example`**) and restart Streamlit/Python so **`papers_rag_config`** reloads.
- **PubMed missing or `ncbi_esearch_error`:** use the in-app sidebar **📄 Extract / refresh abstract_meta** with PubMed checked **or** run **`extract_abstracts.py --pubmed-meta`** from a terminal (see [Section 10](#section-10)); set **NCBI credentials** (**`NCBI_EMAIL`** = your contact address; optional **`NCBI_API_KEY`**); inspect **`pubmed_enrichment.status`** / **`error`** in the JSON; transient NCBI faults may clear on retry.

---

<a id="section-10"></a>
## Section 10: abstract_meta directory: NCBI / PubMed and refresh workflow

Per-PDF JSON lives under your configured **`ABSTRACT_META_ROOT`** (mirroring relative paths below **`PAPERS_DIR`**). **Two ways to write or refresh files:**

1. **In the Streamlit app (typical):** sidebar **📄 Extract / refresh abstract_meta** (**`run_abstract_extractions`**) — same engine as **`extract_abstracts.py`** (scope: only missing vs full refresh; optional PubMed when **`NCBI_EMAIL`** is set). Use this to manage **`abstract_meta/`** without a separate terminal.
2. **Manually / scripted (CLI):** run **`extract_abstracts.py`** from a shell — batch jobs, **`tee`** logs, or flags not exposed in the UI. See **§10.3**.

The running app **reads** these JSON files for titles, abstracts, and export (**it does not** call NCBI during ordinary search — lookups happen when you run the extractor with PubMed enabled).

### 10.1 Fields (conceptual)

- **abstract_text** — extracted from the PDF (always attempted when you run the script).
- **abstract_pubmed** — Medline-style abstract body from **`efetch`**, only when **`--pubmed-meta`** succeeded on the abstract fetch.
- **pubmed_enrichment** — PMID resolution metadata (**`esearch`**, **`esummary`**), **`schema_version`** 3, **`status`** (e.g. **ok**, **no_hit**, **http_error**, **`ncbi_esearch_error`**, …), and **`error`** detail strings when lookups fail.

### 10.2 NCBI credentials (`.env`)

For **`--pubmed-meta`**, supply **NCBI credentials** in **`.env`** (see **`.env.example`**):

- **`NCBI_EMAIL`** — the **contact address** you register with NCBI for E-utilities. This is **not** a special “@ncbi” mailbox: it is usually **your usual email** (Gmail, institutional mail, etc.). Entrez attaches it as the courteous identity string on HTTP requests.
- **`NCBI_API_KEY`** — **optional** key from your NCBI account; raises allowed request throughput. Request / manage via NCBI’s API-key documentation.
- **`ENTREZ_EMAIL`** / **`ENTREZ_API_KEY`** — accepted **aliases** for the two variables above (same values).

### 10.3 Refresh or backfill sidescars (in-app and CLI)

**A) Streamlit sidebar (in-app)** — With the app running, use **📄 Extract / refresh abstract_meta** in the sidebar. This is the usual way after **clone** or when adding PDFs: choose scope (**only missing** vs **full refresh**), toggle **PubMed** when **`NCBI_EMAIL`** is in **`.env`**, and optional **PDF newer than JSON** when only-missing applies. No terminal required for routine work.

**B) Command line (manual, automation, log capture)** — From **`papers-rag_app`** with Conda active:

```bash
conda activate papers_rag
cd /path/to/papers-rag_app
```

**All PDFs** under the configured papers directory (re-extract every file and re-query PubMed when flags allow):

```bash
python extract_abstracts.py --pubmed-meta --force
```

**Only PDFs that have no JSON yet** (fastest for a growing library):

```bash
python extract_abstracts.py --pubmed-meta --only-missing
```

**Same as only-missing, but also re-run when the PDF is newer than its JSON:**

```bash
python extract_abstracts.py --pubmed-meta --only-missing --refresh-if-newer-pdf
```

**Smoke test on the first N files:**

```bash
python extract_abstracts.py --pubmed-meta --limit 20
```

**Different `ABSTRACT_META_ROOT`** than **`.env`** default (advanced):

```bash
python extract_abstracts.py --abstract-meta-root /abs/path/to/meta_mirror --pubmed-meta
```

**Different papers folder** than **`PAPERS_DIR`** from **`.env`**:

```bash
python extract_abstracts.py --papers-dir /path/to/papers/root --pubmed-meta
```

Run **`python extract_abstracts.py -h`** for the full argparse help.

### 10.4 Capturing a full run log (stdout + stderr)

POSIX / bash (**`tee`** keeps terminal output **and** a file):

```bash
python extract_abstracts.py --pubmed-meta 2>&1 | tee extract_abstracts_run.log
```

Use the same pattern with **`--only-missing`**, **`--force`**, etc. **`nohup … &`** is optional for long batch jobs; still append **`2>&1 | tee …`** if you want one combined log file.

On **Windows PowerShell**, a common pattern is:

```powershell
python extract_abstracts.py --pubmed-meta *>&1 | Tee-Object -FilePath extract_abstracts_run.log
```

### 10.5 Repository note

**`abstract_meta/`** is **gitignored** by default (large, machine-local). After **clone**, recreate it with the sidebar control (**§10.3 A**) or the CLI (**§10.3 B**).

---

*End — Papers RAG installation & operations guide (v2.5, public / sanitized).*
