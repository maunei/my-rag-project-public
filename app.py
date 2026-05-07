"""
app.py — Papers RAG: Semantic Search & Deep Chat over your PDF library.

Run with:
    conda activate papers_rag
    streamlit run app.py
"""

import shutil
from collections import defaultdict
from pathlib import Path

import streamlit as st
from fastembed import TextEmbedding

from indexer import (
    index_papers,
    is_indexed,
    get_index_stats,
    DB_PATH,
    PAPERS_DIR,
    EMBEDDING_MODEL,
    semantic_search,
    hybrid_search,
)
from rag_engine import (
    get_gemini_client,
    upload_pdfs_to_gcs,
    stream_pdf_chat,
    stream_rag_response,
)
from pdf_server import start_pdf_server, pdf_url, PDF_SERVER_PORT

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Papers RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Start PDF file server (once per process) ──────────────────────────────────

@st.cache_resource
def _start_file_server():
    start_pdf_server(papers_dir=PAPERS_DIR, port=PDF_SERVER_PORT)
    return True

_start_file_server()

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to Gemini…")
def load_gemini_client():
    try:
        return get_gemini_client()
    except Exception as e:
        st.error(f"Gemini connection failed: {e}")
        st.stop()


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "search_results": [],       # all retrieved hits (unfiltered)
        "last_search_query": "",
        "quick_chat_history": [],   # [{role, content}] for Quick Chat (excerpt-based)
        "deep_chat_papers": [],     # file_paths loaded into Deep Chat
        "deep_chat_history": [],    # [{role, content}] for Deep Chat
        "gemini_uploads": {},       # {file_path: gs://URI or None} from GCS
        "indexing_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _score_color(score: float) -> str:
    pct = int(score * 100)
    if pct >= 75:
        return "green"
    if pct >= 55:
        return "orange"
    return "red"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Papers RAG")
    st.caption("Semantic Search & Deep Chat over your PDF library")
    st.divider()

    # ── Index status & controls ───────────────────────────────────────────────
    indexed = is_indexed()
    if indexed:
        stats = get_index_stats()
        st.success(
            f"✅ Index ready\n\n"
            f"**{stats['total_papers']:,}** papers · "
            f"**{stats['total_chunks']:,}** chunks"
        )
    else:
        st.warning("⚠️ Papers not indexed yet")

    btn_label = "🔄 Update Index" if indexed else "⚙️ Build Index (first run)"
    if st.button(btn_label, use_container_width=True):
        embed_model = load_embedding_model()
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress(frac, msg):
            progress_bar.progress(frac)
            status_text.caption(msg)

        with st.spinner("Indexing…"):
            result = index_papers(
                embedding_model=embed_model,
                progress_callback=_progress,
            )
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Done! Indexed **{result['indexed']}** · "
            f"Skipped **{result['skipped']}** · "
            f"Errors **{result['errors']}** · "
            f"Chunks **{result['total_chunks']:,}**"
        )
        st.session_state["indexing_done"] = True
        st.rerun()

    st.divider()
    st.caption(f"Papers: `{PAPERS_DIR}`")
    st.caption(f"Index: `{DB_PATH}`")
    st.caption(f"PDF server: `http://localhost:{PDF_SERVER_PORT}`")


# ── Main area ─────────────────────────────────────────────────────────────────

if not is_indexed():
    st.title("📚 Papers RAG")
    st.info(
        "👈 Click **Build Index** in the sidebar to get started.\n\n"
        "This scans all PDFs, extracts text, generates embeddings, and stores "
        "them locally. It runs **once** (~10–30 min for 833 PDFs) and updates "
        "incrementally after that."
    )
    st.stop()

embed_model   = load_embedding_model()
gemini_client = load_gemini_client()

tab_search, tab_chat = st.tabs(["🔍 Semantic Search", "💬 Deep Chat with Gemini"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Semantic Search
# ════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔍 Semantic Search")

    # ── Search controls (inside a form so nothing reruns until Submit) ─────────
    with st.form("search_form"):
        col_q, col_cut, col_max = st.columns([5, 1.2, 1.2])
        with col_q:
            search_query = st.text_input(
                "Query",
                placeholder="e.g. spinal cord interneuron development ATAC-seq",
                label_visibility="collapsed",
            )
        with col_cut:
            cutoff = st.number_input(
                "Min similarity",
                min_value=0.0, max_value=1.0,
                value=0.85, step=0.01,
                format="%.2f",
                help="Only show results with similarity ≥ this value (0–1)",
            )
        with col_max:
            max_str = st.text_input(
                "Max results",
                value="",
                placeholder="∞",
                help="Leave empty for no limit (show all above cutoff)",
            )
            try:
                max_results = int(max_str) if max_str.strip() else None
            except ValueError:
                max_results = None

        search_btn = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    if search_btn and search_query:
        with st.spinner("Searching…"):
            # Hybrid: semantic search + keyword fallback for tool/gene names
            raw_hits = hybrid_search(
                query=search_query,
                embedding_model=embed_model,
                db_path=DB_PATH,
                n_results=500,
            )
            st.session_state["search_results"] = raw_hits
            st.session_state["last_search_query"] = search_query
            st.session_state["quick_chat_history"] = []  # reset on new search

    raw_results = st.session_state.get("search_results", [])

    if raw_results:
        # Apply similarity cutoff; keyword hits bypass it so tool/gene names always show
        filtered = [
            h for h in raw_results
            if h.get("match_type") == "keyword" or h["score"] >= cutoff
        ]

        # Group by paper (file_path) — one card per unique paper
        papers_map: dict[str, list] = defaultdict(list)
        for hit in filtered:
            papers_map[hit["metadata"]["file_path"]].append(hit)

        # Sort papers by their best (highest) chunk score
        sorted_papers = sorted(
            papers_map.items(),
            key=lambda kv: max(h["score"] for h in kv[1]),
            reverse=True,
        )

        # Apply max_results cap (per unique paper)
        if max_results:
            sorted_papers = sorted_papers[:max_results]

        query_label = st.session_state["last_search_query"]
        st.subheader(
            f"Found **{len(sorted_papers)}** paper(s) above {int(cutoff*100)}% similarity"
            f" for: *{query_label}*"
        )

        if not sorted_papers:
            st.info(
                f"No results above {int(cutoff*100)}% similarity. "
                "Try lowering the cutoff or rephrasing your query."
            )
        else:
            # ── Result cards (display only — no widgets, zero rerun cost) ─────
            for fp, hits in sorted_papers:
                best_score = max(h["score"] for h in hits)
                m0 = hits[0]["metadata"]
                score_pct = int(best_score * 100)
                color = _score_color(best_score)
                pdf_link = pdf_url(fp, PAPERS_DIR)

                # Detect whether any hit came only from keyword match
                has_keyword = any(h.get("match_type") == "keyword" for h in hits)
                has_semantic = any(h.get("match_type") == "semantic" for h in hits)
                if has_keyword and not has_semantic:
                    match_badge = " · 🔑 keyword match"
                elif has_keyword:
                    match_badge = " · 🔑 +keyword"
                else:
                    match_badge = ""

                title_link = (
                    f'<a href="{pdf_link}" target="_blank" '
                    f'style="font-size:1.05em; font-weight:600; '
                    f'text-decoration:none;">'
                    f'📄 {m0["paper_title"]}</a>'
                )
                st.markdown(title_link, unsafe_allow_html=True)
                st.markdown(
                    f':{color}[**{score_pct}% similarity**] · '
                    f'`{Path(fp).name}` · '
                    f'{len(hits)} matching chunk(s){match_badge}',
                    unsafe_allow_html=False,
                )

                with st.expander(f"Show {len(hits)} matching excerpt(s)", expanded=False):
                    for hit in sorted(hits, key=lambda h: h["score"], reverse=True):
                        hm = hit["metadata"]
                        hp = int(hit["score"] * 100)
                        hit_badge = " · 🔑 keyword" if hit.get("match_type") == "keyword" else ""
                        st.markdown(
                            f"**Page {hm['page_num']}** — "
                            f":{_score_color(hit['score'])}[{hp}% similarity]{hit_badge}"
                        )
                        st.write(hit["text"])
                        st.markdown("---")

            # ── Quick Chat (excerpt-based, no upload needed) ───────────────────
            st.divider()
            st.subheader("💬 Quick Chat (based on search excerpts)")
            st.caption(
                "Ask Gemini questions using only the excerpts found above — "
                "no upload needed. Use this to explore the results and decide "
                "which papers are worth uploading to Deep Chat."
            )

            # Build one best-scoring chunk per paper from the already-retrieved
            # search results so ALL papers are always represented in Quick Chat,
            # regardless of the MAX_CONTEXT_CHUNKS limit.
            quick_context_chunks: list[dict] = [
                max(hits, key=lambda h: h["score"])
                for _, hits in sorted_papers
            ]

            quick_history: list[dict] = st.session_state.get("quick_chat_history", [])

            # Capture input first so new messages always appear after history
            quick_input = st.chat_input(
                "Ask about the search results… e.g. What are the main methods used?",
                key="quick_chat_input",
            )

            if quick_history:
                col_qc, col_qcbtn = st.columns([4, 1])
                with col_qcbtn:
                    if st.button("🗑 Clear", key="clear_quick_chat", use_container_width=True):
                        st.session_state["quick_chat_history"] = []
                        st.rerun()

            for msg in quick_history:
                role_display = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role_display):
                    st.markdown(msg["content"])

            if quick_input:
                with st.chat_message("user"):
                    st.markdown(quick_input)

                st.session_state["quick_chat_history"].append(
                    {"role": "user", "content": quick_input}
                )

                with st.chat_message("assistant"):
                    placeholder_qc = st.empty()
                    full_qc_response = ""
                    prior_qc_history = st.session_state["quick_chat_history"][:-1]

                    try:
                        for text_chunk, sources in stream_rag_response(
                            query=quick_input,
                            chat_history=prior_qc_history,
                            embedding_model=embed_model,
                            gemini_client=gemini_client,
                            preloaded_chunks=quick_context_chunks,
                        ):
                            if sources is not None:
                                continue  # sources already visible as result cards above
                            if text_chunk:
                                full_qc_response += text_chunk
                                placeholder_qc.markdown(full_qc_response + "▌")

                        placeholder_qc.markdown(full_qc_response)

                    except Exception as e:
                        full_qc_response = f"⚠️ Error: {e}"
                        placeholder_qc.error(full_qc_response)

                st.session_state["quick_chat_history"].append(
                    {"role": "model", "content": full_qc_response}
                )

            # ── Paper selection form (no rerun until Send is clicked) ─────────
            st.divider()
            st.markdown("**Select papers to send to Deep Chat** *(check/uncheck freely — nothing runs until you click Send):*")

            with st.form("send_to_chat_form"):
                fp_keys: list[tuple[str, str]] = []
                for fp, hits in sorted_papers:
                    best_score = max(h["score"] for h in hits)
                    score_pct = int(best_score * 100)
                    color = _score_color(best_score)
                    fname = Path(fp).name
                    checked = st.checkbox(
                        f":{color}[**{score_pct}%**]  {fname}",
                        value=True,
                        key=f"sel_{fp}",
                    )
                    fp_keys.append((fp, checked))

                submitted = st.form_submit_button(
                    "💬 Send selected papers to Deep Chat →",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                selected_fps = [fp for fp, checked in fp_keys if checked]
                if selected_fps:
                    st.session_state["deep_chat_papers"] = selected_fps
                    st.session_state["deep_chat_history"] = []
                    st.session_state["gemini_uploads"] = {}  # reset uploads for new selection

                    # Copy selected PDFs to selected_pdfs/ folder
                    selected_dir = Path(__file__).parent / "selected_pdfs"
                    selected_dir.mkdir(exist_ok=True)
                    # Clear previous contents so folder always reflects current selection
                    for old in selected_dir.glob("*.pdf"):
                        old.unlink()
                    for fp in selected_fps:
                        shutil.copy2(fp, selected_dir / Path(fp).name)

                    st.success(
                        f"✅ {len(selected_fps)} paper(s) loaded into Deep Chat "
                        f"and copied to `selected_pdfs/`. "
                        "Switch to the **💬 Deep Chat with Gemini** tab."
                    )
                else:
                    st.warning("No papers selected — check at least one box.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Chat with Gemini (full PDFs)
# ════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("💬 Deep Chat with Gemini")
    st.caption(
        "Gemini reads the **full content** of the selected PDFs. "
        "Upload them first so Gemini can see all papers without size limits."
    )

    deep_papers = st.session_state.get("deep_chat_papers", [])
    deep_history = st.session_state.get("deep_chat_history", [])
    gemini_uploads: dict = st.session_state.get("gemini_uploads", {})

    # ── No papers loaded ──────────────────────────────────────────────────────
    if not deep_papers:
        st.info(
            "No papers loaded yet.\n\n"
            "**How to add papers:**\n"
            "Use the **🔍 Semantic Search** tab → run a search → "
            "select papers at the bottom → click *Send to Deep Chat*"
        )
    else:
        # ── Paper list + upload status ────────────────────────────────────────
        uploaded_count = sum(1 for fp in deep_papers if gemini_uploads.get(fp) is not None)
        failed_count   = sum(1 for fp in deep_papers if fp in gemini_uploads and gemini_uploads[fp] is None)
        pending_count  = len(deep_papers) - uploaded_count - failed_count

        with st.expander(
            f"📚 {len(deep_papers)} paper(s) · "
            f"{'✅ All uploaded' if uploaded_count == len(deep_papers) else f'⬆ {pending_count} pending · ✅ {uploaded_count} uploaded' + (f' · ❌ {failed_count} failed' if failed_count else '')}",
            expanded=(uploaded_count < len(deep_papers)),
        ):
            for fp in deep_papers:
                name = Path(fp).name
                link = pdf_url(fp, PAPERS_DIR)
                if gemini_uploads.get(fp) is not None:
                    badge = "✅"
                elif fp in gemini_uploads:
                    badge = "❌"
                else:
                    badge = "⏳"
                st.markdown(
                    f'{badge} <a href="{link}" target="_blank">{name}</a>',
                    unsafe_allow_html=True,
                )

        # ── Upload button ─────────────────────────────────────────────────────
        all_uploaded = uploaded_count == len(deep_papers)

        if not all_uploaded:
            col_up, col_rst = st.columns([3, 1])
            with col_up:
                upload_btn = st.button(
                    f"⬆ Upload {len(deep_papers)} paper(s) to Google Cloud",
                    type="primary",
                    use_container_width=True,
                    help="PDFs are uploaded to Google Cloud Storage and read by Gemini via gs:// URI. No size limits.",
                )
            with col_rst:
                if st.button("🗑 Clear", use_container_width=True):
                    st.session_state["deep_chat_papers"] = []
                    st.session_state["deep_chat_history"] = []
                    st.session_state["gemini_uploads"] = {}
                    st.rerun()

            if upload_btn:
                status_box = st.empty()
                progress_bar = st.progress(0.0)
                upload_results: dict = {}

                def _on_progress(current, total, name, success, error):
                    frac = current / total
                    progress_bar.progress(frac)
                    icon = "✅" if success else "❌"
                    msg = f"{icon} ({current}/{total}) {name}"
                    if not success:
                        msg += f" — {error}"
                    status_box.info(msg)
                    upload_results.update({})  # just to capture closure

                results = upload_pdfs_to_gcs(
                    pdf_paths=deep_papers,
                    progress_callback=_on_progress,
                )
                st.session_state["gemini_uploads"] = results
                progress_bar.empty()
                status_box.empty()
                st.rerun()
        else:
            col_info, col_rst = st.columns([3, 1])
            with col_info:
                st.success(f"✅ All {len(deep_papers)} paper(s) uploaded — ready to chat.")
            with col_rst:
                if st.button("🗑 Clear", use_container_width=True):
                    st.session_state["deep_chat_papers"] = []
                    st.session_state["deep_chat_history"] = []
                    st.session_state["gemini_uploads"] = {}
                    st.rerun()

        # ── Conversation history & chat ───────────────────────────────────────
        if all_uploaded:
            st.divider()

            # Capture input before rendering history so new messages always
            # appear after existing ones (chat_input floats to bottom of page).
            user_input = st.chat_input(
                "Ask anything about the uploaded papers… "
                "e.g. Summarize the main findings / What methods were used?"
            )

            if deep_history:
                col_hist, col_turns = st.columns([3, 1])
                with col_turns:
                    turns = len([m for m in deep_history if m["role"] == "user"])
                    st.caption(f"{turns} turn(s)")
                if st.button("🗑 Clear conversation", use_container_width=False):
                    st.session_state["deep_chat_history"] = []
                    st.rerun()

            for msg in deep_history:
                role_display = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role_display):
                    st.markdown(msg["content"])

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)

                st.session_state["deep_chat_history"].append(
                    {"role": "user", "content": user_input}
                )

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    prior_history = st.session_state["deep_chat_history"][:-1]

                    try:
                        for chunk in stream_pdf_chat(
                            query=user_input,
                            pdf_paths=deep_papers,
                            chat_history=prior_history,
                            gemini_client=gemini_client,
                            uploaded_files=st.session_state["gemini_uploads"],
                        ):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")

                        placeholder.markdown(full_response)

                    except Exception as e:
                        full_response = f"⚠️ Error: {e}"
                        placeholder.error(full_response)

                st.session_state["deep_chat_history"].append(
                    {"role": "model", "content": full_response}
                )
        else:
            st.info("⬆ Upload the papers to Gemini first, then the chat will appear.")
