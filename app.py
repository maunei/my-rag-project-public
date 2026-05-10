"""
app.py — Papers RAG: Semantic Search & Deep Chat over your PDF library.

Run with:
    conda activate papers_rag
    streamlit run app.py
"""

import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException
from fastembed import TextEmbedding

from indexer import (
    MAX_BOOLEAN_CLAUSES,
    boolean_retrieval_segmented,
    format_boolean_expression_preview,
    format_boolean_expression_translation_md,
    get_index_stats,
    get_indexed_papers,
    index_papers,
    is_indexed,
    DB_PATH,
    PAPERS_DIR,
    EMBEDDING_MODEL,
)
from abstract_extraction import (
    ABSTRACT_META_ROOT,
    ensure_export_dir,
    load_abstract_record,
)
from rag_engine import (
    build_external_llm_context_text,
    get_gemini_client,
    upload_pdfs_to_gcs,
    stream_pdf_chat,
    stream_rag_response,
)
from pdf_server import start_pdf_server, pdf_url, PDF_SERVER_PORT


def _streamlit_supports_fragment() -> bool:
    """Partial reruns for checkbox tweaks require Streamlit 1.37+ (st.fragment)."""
    if not hasattr(st, "fragment"):
        return False
    try:
        parts = st.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (1, 37)
    except (ValueError, IndexError):
        return True


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Papers RAG V2.3",
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
        "last_search_cutoff": 0.6,
        "last_search_max_results": None,  # int | None — cap from last successful search
        "last_search_max_results_str": "",
        "manual_extra_fps": [],     # pasted / added paths not requiring a search
        "bool_clause_count": 1,
        "bool_group_splits": [],    # clause indices: new group starts after this clause
        "quick_chat_history": [],   # [{role, content}] for Quick Chat (excerpt-based)
        "deep_chat_papers": [],     # file_paths loaded into Deep Chat
        "deep_chat_history": [],    # [{role, content}] for Deep Chat
        "gemini_uploads": {},       # {file_path: gs://URI or None} from GCS
        "indexing_done": False,
        "_index_stats_cache_gen": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sanitize_bool_splits(n_clauses: int, splits: list) -> list[int]:
    """Keep only valid split indices (boundary after clause ``s``); ``0 <= s <= n-2``."""
    if n_clauses < 2:
        return []
    max_s = n_clauses - 2
    try:
        out = sorted({int(s) for s in splits if 0 <= int(s) <= max_s})
    except (TypeError, ValueError):
        return []
    return out


def _normalize_paste_line(raw_line: str) -> str:
    """Strip markdown bullets and wrapping backticks from pasted basename/path lines."""
    s = raw_line.strip()
    if not s:
        return ""
    s = re.sub(r"^\d+\.\s+", "", s)
    s = re.sub(r"^[\-\*]\s+", "", s)
    return s.strip("`").strip()


def _compute_filtered_papers(
    raw_results: list,
    cutoff: float,
    max_results: int | None,
) -> tuple[list[tuple[str, list]], dict[str, list]]:
    """Same filtering/sorting as Tab 1 hit list; returns (sorted_papers, papers_map)."""
    papers_map: dict[str, list] = defaultdict(list)
    if not raw_results:
        return [], papers_map
    filtered = [
        h for h in raw_results
        if h.get("match_type") == "keyword" or h["score"] >= cutoff
    ]
    for hit in filtered:
        papers_map[hit["metadata"]["file_path"]].append(hit)
    sorted_papers = sorted(
        papers_map.items(),
        key=lambda kv: max(h["score"] for h in kv[1]),
        reverse=True,
    )
    if max_results:
        sorted_papers = sorted_papers[:max_results]
    return sorted_papers, papers_map


def _stage_deep_chat_pdfs(source_pdf_paths: list[str]) -> None:
    """Stage PDFs under ``selected_pdfs/`` via symlinks (fast); copy if symlinks fail."""
    base = Path(__file__).resolve().parent / "selected_pdfs"
    base.mkdir(parents=True, exist_ok=True)
    for p in base.iterdir():
        try:
            p.unlink()
        except OSError:
            pass
    for fp in source_pdf_paths:
        dest = base / Path(fp).name
        src = Path(fp).expanduser().resolve()
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src)
        except OSError:
            shutil.copy2(src, dest)


@st.cache_data(ttl=120)
def _cached_index_stats(cache_generation: int) -> dict:
    """Invalidate by bumping ``cache_generation`` in session state after re-indexing."""
    return get_index_stats()


def _clear_manual_add_callback() -> None:
    prev = list(st.session_state.get("manual_extra_fps") or [])
    st.session_state["manual_extra_fps"] = []
    if "paste_basenames" in st.session_state:
        st.session_state["paste_basenames"] = ""

    cutoff = float(
        st.session_state.get(
            "_tab1_cutoff_live",
            st.session_state.get("last_search_cutoff", 0.6),
        )
    )
    max_results = st.session_state.get("_tab1_max_results_live")
    raw = st.session_state.get("search_results", [])
    sorted_papers, _ = _compute_filtered_papers(raw, cutoff, max_results)
    hit_fps = {fp for fp, _ in sorted_papers}

    # Only drop selection for paths that were manual-only. Papers that also appear
    # in the current hit list keep ``sel_*`` so checkboxes / Export stay consistent.
    for fp in prev:
        if fp not in hit_fps:
            st.session_state[f"sel_{fp}"] = False


def _clear_tab1_all_callback() -> None:
    _clear_manual_add_callback()
    st.session_state["search_results"] = []
    st.session_state["last_search_query"] = ""
    st.session_state["quick_chat_history"] = []
    st.session_state["last_search_max_results_str"] = ""
    st.session_state["bool_group_splits"] = []
    for k in list(st.session_state.keys()):
        if k.startswith("sel_"):
            del st.session_state[k]


def _make_bulk_selection_callback(hit_fps: tuple[str, ...], manual_fps: tuple[str, ...], value: bool):
    def _cb() -> None:
        for fp in hit_fps:
            st.session_state[f"sel_{fp}"] = value
        for fp in manual_fps:
            st.session_state[f"sel_{fp}"] = value

    return _cb


def _score_color(score: float) -> str:
    pct = int(score * 100)
    if pct >= 75:
        return "green"
    if pct >= 55:
        return "orange"
    return "red"


@st.cache_data(ttl=120)
def _basename_resolve_map() -> dict[str, list[str]]:
    """casefold basename -> list of indexed absolute paths (may be ambiguous)."""
    d: dict[str, list[str]] = defaultdict(list)
    for p in get_indexed_papers():
        k = Path(p["file_path"]).name.casefold()
        d[k].append(p["file_path"])
    return {k: sorted(set(v)) for k, v in d.items()}


def _abstract_payload(fp: str) -> dict:
    rec = load_abstract_record(fp, PAPERS_DIR)
    if rec:
        return rec
    msg = "Abstract not detected."
    return {
        "file_name": Path(fp).name,
        "abstract_text": msg,
        "status": "missing_json",
        "source": "none",
        "char_count": len(msg),
        "word_count": len(msg.split()),
    }


def _flush_deep_chat_staged_banner() -> None:
    """Show one-shot list of PDFs staged for Deep Chat (survives ``st.rerun()``)."""
    payload = st.session_state.pop("_deep_chat_staged_banner", None)
    if not payload:
        return
    names = list(payload.get("names") or [])
    n = int(payload.get("count", len(names)))
    cap = 40
    lines = "\n".join(f"- `{x}`" for x in names[:cap])
    if len(names) > cap:
        lines += f"\n\n_(…and {len(names) - cap} more)_"
    st.success(
        f"✅ **{n}** paper(s) staged for Deep Chat in `selected_pdfs/` "
        "(symlinks when supported — otherwise copied). "
        "Open **💬 Deep Chat with Gemini** to upload to Gemini.\n\n"
        + (lines if lines else "_(no filenames recorded)_")
    )


def _render_paper_selection_widgets(
    sorted_papers: list[tuple[str, list]],
    manual_only: list[str],
    papers_map: dict[str, list],
    embed_model,
    gemini_client,
) -> None:
    """Selection lists + Export / Deep Chat / Quick Chat (same st.fragment as checkboxes)."""
    has_hits = bool(sorted_papers)
    has_manual = bool(manual_only)
    if not has_hits and not has_manual:
        return

    hit_fps_t = tuple(fp for fp, _ in sorted_papers)
    man_fps_t = tuple(manual_only)

    st.divider()
    st.markdown(
        "**Select papers** — Quick Chat & export use **checked** rows; "
        "**Send to Deep Chat** loads all checked PDFs."
    )

    col_sa, col_da = st.columns(2)
    with col_sa:
        st.button(
            "✅ Select all",
            key="quick_sel_all",
            use_container_width=True,
            on_click=_make_bulk_selection_callback(hit_fps_t, man_fps_t, True),
        )
    with col_da:
        st.button(
            "⬜ Deselect all",
            key="quick_desel_all",
            use_container_width=True,
            on_click=_make_bulk_selection_callback(hit_fps_t, man_fps_t, False),
        )

    if has_manual:
        nm = len(manual_only)
        with st.expander(
            f"📂 Manual add (not in current search hits) — **{nm}** paper(s)",
            expanded=False,
        ):
            st.caption("**abstract_meta** JSON only — no search excerpts.")
            for fp in sorted(manual_only, key=lambda p: Path(p).name.casefold()):
                pdf_link = pdf_url(fp, PAPERS_DIR)
                name = Path(fp).name
                st.markdown(
                    f'`manual` · <a href="{pdf_link}" target="_blank"><b>{name}</b></a>',
                    unsafe_allow_html=True,
                )
                ap = _abstract_payload(fp)
                with st.expander("📋 Abstract", expanded=False):
                    st.caption(
                        f"status: **{ap['status']}** · source: `{ap['source']}` · "
                        f"{ap.get('char_count', '—')} chars · {ap.get('word_count', '—')} words"
                    )
                    st.write(ap.get("abstract_text", ""))
                st.checkbox(f"Select `{name}`", key=f"sel_{fp}")

    if has_hits:
        nh = len(sorted_papers)
        with st.expander(
            f"📂 Search hit papers — **{nh}** paper(s)",
            expanded=False,
        ):
            for fp, hits in sorted_papers:
                best_score = max(h["score"] for h in hits)
                score_pct = int(best_score * 100)
                color = _score_color(best_score)
                fname = Path(fp).name
                st.checkbox(
                    f":{color}[**{score_pct}%**]  {fname}",
                    key=f"sel_{fp}",
                )

    ordered_checked: list[str] = []
    seen_oc: set[str] = set()
    for fp, _ in sorted_papers:
        if st.session_state.get(f"sel_{fp}", False) and fp not in seen_oc:
            ordered_checked.append(fp)
            seen_oc.add(fp)
    for fp in sorted(manual_only, key=lambda p: Path(p).name.casefold()):
        if st.session_state.get(f"sel_{fp}", False) and fp not in seen_oc:
            ordered_checked.append(fp)
            seen_oc.add(fp)

    has_selection = bool(ordered_checked)
    if not has_selection:
        st.caption(
            "⚠️ **No papers checked** — use the lists above (expand if folded), "
            "then check at least one row to enable **Export** and **Deep Chat**."
        )

    exp_help = (
        "Abstract JSON + excerpts for checked papers (abstract-only if no hits)."
        if has_selection
        else "Select at least one paper above first."
    )
    deep_help = (
        "Stage PDF paths for Deep Chat (symlinks under selected_pdfs/)."
        if has_selection
        else "Select at least one paper above first."
    )

    export_clicked = st.button(
        "📄 Export context for external LLM",
        use_container_width=True,
        disabled=not has_selection,
        help=exp_help,
    )
    deep_send = st.button(
        "💬 Send selected papers to Deep Chat →",
        type="primary",
        use_container_width=True,
        disabled=not has_selection,
        help=deep_help,
    )

    if export_clicked and ordered_checked:
        export_chunks: list[dict] = []
        abstracts_export: dict[str, dict] = {}
        for fp in ordered_checked:
            abstracts_export[fp] = _abstract_payload(fp)
            if fp in papers_map:
                export_chunks.extend(papers_map[fp])
        body = build_external_llm_context_text(
            export_chunks, abstracts_export, ordered_checked
        )
        out_dir = ensure_export_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"prompt_context_{ts}.txt"
        out_path.write_text(body, encoding="utf-8")
        st.success(f"Saved **`{out_path}`** ({len(ordered_checked)} paper(s)).")

    if deep_send and ordered_checked:
        st.session_state["deep_chat_papers"] = ordered_checked
        st.session_state["deep_chat_history"] = []
        st.session_state["gemini_uploads"] = {}

        _stage_deep_chat_pdfs(ordered_checked)

        st.session_state["_deep_chat_staged_banner"] = {
            "count": len(ordered_checked),
            "names": [Path(fp).name for fp in ordered_checked],
        }
        st.rerun()  # fragment-only runs skip ``tab_chat``; full rerun refreshes Tab 2.

    st.divider()
    st.subheader("💬 Quick Chat (Vertex)")
    st.caption(
        "Checked papers only: **abstract JSON** for each; **best excerpt** when the "
        "paper came from search hits. Filename citations required."
    )

    quick_history: list[dict] = st.session_state.get("quick_chat_history", [])
    quick_input = st.chat_input(
        "Ask about the checked papers…",
        key="quick_chat_input",
    )

    if quick_history:
        _, col_qcbtn = st.columns([4, 1])
        with col_qcbtn:
            if st.button("🗑 Clear", key="clear_quick_chat", use_container_width=True):
                st.session_state["quick_chat_history"] = []
                try:
                    st.rerun(scope="fragment")
                except StreamlitAPIException:
                    st.rerun()

    for msg in quick_history:
        role_display = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role_display):
            st.markdown(msg["content"])

    if quick_input:
        if not ordered_checked:
            with st.chat_message("assistant"):
                st.error("Select at least one paper before using Quick Chat.")
        else:
            quick_context_chunks: list[dict] = []
            for fp in ordered_checked:
                if fp in papers_map:
                    hlist = papers_map[fp]
                    quick_context_chunks.append(max(hlist, key=lambda h: h["score"]))

            abstracts_quick: dict[str, dict] = {}
            for fp in ordered_checked:
                abstracts_quick[fp] = _abstract_payload(fp)

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
                        preloaded_chunks=quick_context_chunks if quick_context_chunks else [],
                        abstracts_by_file_path=abstracts_quick,
                    ):
                        if sources is not None:
                            continue
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Papers RAG V2.3")
    st.caption("Semantic Search & Deep Chat over your PDF library")
    st.divider()

    # ── Index status & controls ───────────────────────────────────────────────
    indexed = is_indexed()
    if indexed:
        stats = _cached_index_stats(int(st.session_state.get("_index_stats_cache_gen", 0)))
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
        st.session_state["_index_stats_cache_gen"] = (
            int(st.session_state.get("_index_stats_cache_gen", 0)) + 1
        )
        try:
            _basename_resolve_map.clear()
        except Exception:
            pass
        st.rerun()

    st.divider()
    st.caption(f"Papers: `{PAPERS_DIR}`")
    st.caption(f"Index: `{DB_PATH}`")
    st.caption(f"Abstract meta: `{ABSTRACT_META_ROOT}`")
    st.caption(f"PDF server: `http://localhost:{PDF_SERVER_PORT}`")


# ── Main area ─────────────────────────────────────────────────────────────────

if not is_indexed():
    st.title("📚 Papers RAG V2.3")
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
    _flush_deep_chat_staged_banner()

    st.subheader("Paste PDF basenames")
    st.caption(
        "One filename per line (no paths required), **case-insensitive**. "
        "Resolves against the **indexed** corpus. Works **without** running a search."
    )
    st.text_area(
        "Paste basenames",
        height=120,
        key="paste_basenames",
        placeholder="paper_one.pdf\npaper_two.pdf",
        label_visibility="collapsed",
    )
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        apply_paste = st.button("Apply pasted names", key="apply_paste_btn")
    with pc2:
        st.button(
            "Clear manual-add list",
            key="clear_manual_btn",
            use_container_width=True,
            on_click=_clear_manual_add_callback,
        )
    with pc3:
        st.button(
            "Clear all (search, paste, selections)",
            key="clear_tab1_all_btn",
            use_container_width=True,
            on_click=_clear_tab1_all_callback,
        )

    if apply_paste:
        blob = st.session_state.get("paste_basenames") or ""
        bmap = _basename_resolve_map()
        matched_new = []
        unknown: list[str] = []
        ambiguous: list[tuple[str, list[str]]] = []
        for raw_line in blob.splitlines():
            line = _normalize_paste_line(raw_line)
            if not line:
                continue
            key = Path(line).name.casefold()
            opts = bmap.get(key)
            if not opts:
                unknown.append(line)
            elif len(opts) > 1:
                ambiguous.append((line, opts))
            else:
                matched_new.append(opts[0])

        matched_new = list(dict.fromkeys(matched_new))
        cutoff_ap = float(st.session_state.get("last_search_cutoff", 0.6))
        max_ap = st.session_state.get("last_search_max_results")
        sp_ap, _ = _compute_filtered_papers(
            st.session_state.get("search_results", []),
            cutoff_ap,
            max_ap,
        )
        hit_fps_ap = {fp for fp, _ in sp_ap}
        in_hits = [fp for fp in matched_new if fp in hit_fps_ap]
        manual_side = [fp for fp in matched_new if fp not in hit_fps_ap]

        cur = list(st.session_state.get("manual_extra_fps") or [])
        cur_set = set(cur)
        for fp in matched_new:
            if fp not in cur_set:
                cur.append(fp)
                cur_set.add(fp)
        for fp in matched_new:
            st.session_state[f"sel_{fp}"] = True
        st.session_state["manual_extra_fps"] = cur

        lines_out = []
        if matched_new:
            lines_out.append(f"Resolved **{len(matched_new)}** name(s) into the manual-add list.")
            if in_hits:
                ih_lines = "\n".join(f"- `{Path(fp).name}`" for fp in in_hits[:40])
                ih_more = (
                    f"\n\n_(…and {len(in_hits) - 40} more)_" if len(in_hits) > 40 else ""
                )
                lines_out.append(
                    f"**In current hit list** ({len(in_hits)}) — checkboxes under "
                    f"*Search hit papers*:\n\n{ih_lines}{ih_more}"
                )
            if manual_side:
                ms_lines = "\n".join(f"- `{Path(fp).name}`" for fp in manual_side[:40])
                ms_more = (
                    f"\n\n_(…and {len(manual_side) - 40} more)_"
                    if len(manual_side) > 40
                    else ""
                )
                lines_out.append(
                    f"**Manual-only** (not in current filtered hits) ({len(manual_side)}):\n\n"
                    f"{ms_lines}{ms_more}"
                )
        if unknown:
            unk_block = "\n".join(f"- `{u}`" for u in unknown[:25])
            more = f"\n\n_(…and {len(unknown) - 25} more)_" if len(unknown) > 25 else ""
            lines_out.append(f"**Unknown** (not in index):\n\n{unk_block}{more}")
        if ambiguous:
            amb_block = "\n".join(
                f"- `{name}` → {len(paths)} paths (skipped)" for name, paths in ambiguous[:15]
            )
            lines_out.append(f"**Ambiguous** basename — skipped:\n\n{amb_block}")
        if lines_out:
            st.success("\n\n".join(lines_out))

    def _boolean_search_panel_fragment() -> None:
        st.divider()
        st.markdown("### Boolean search (**Clause** rows · groups · AND / OR / NOT)")
        st.caption(
            f"Up to **{MAX_BOOLEAN_CLAUSES}** rows (**Clause 1**, **Clause 2**, …). "
            "Optional **groups**: choose where a **new group** starts **after** a Clause; "
            "operators on those edges combine whole groups. "
            "Each row is **semantic** or **keyword** (case-insensitive literal). "
            "**Keyword** Clause rows bypass the similarity cutoff below."
        )

        def _incr_bc() -> None:
            if st.session_state["bool_clause_count"] < MAX_BOOLEAN_CLAUSES:
                st.session_state["bool_clause_count"] += 1

        def _decr_bc() -> None:
            if st.session_state["bool_clause_count"] > 1:
                st.session_state["bool_clause_count"] -= 1
                n = int(st.session_state["bool_clause_count"])
                st.session_state["bool_group_splits"] = _sanitize_bool_splits(
                    n, list(st.session_state.get("bool_group_splits") or [])
                )

        ac1, ac2 = st.columns(2)
        with ac1:
            st.button("Add clause", key="add_clause_btn", on_click=_incr_bc)
        with ac2:
            st.button("Remove last clause", key="remove_clause_btn", on_click=_decr_bc)

        n_bc = int(st.session_state["bool_clause_count"])
        splits_raw = list(st.session_state.get("bool_group_splits") or [])
        splits_sane = _sanitize_bool_splits(n_bc, splits_raw)
        if splits_sane != splits_raw:
            st.session_state["bool_group_splits"] = splits_sane

        if n_bc >= 2:
            st.multiselect(
                "Start a **new group** after Clause…",
                options=list(range(n_bc - 1)),
                format_func=lambda j: f"After Clause {j + 1}",
                key="bool_group_splits",
                help=(
                    "Each contiguous run of **Clause** rows is one parenthesized group "
                    "in the Boolean query. Operators on those edges combine **groups**; "
                    "other operators combine **Clause** rows **within** a group."
                ),
            )

        splits_set = set(
            _sanitize_bool_splits(n_bc, st.session_state.get("bool_group_splits") or [])
        )

        for i in range(n_bc):
            st.markdown(f"**Clause {i + 1}**")
            row1, row2 = st.columns([4, 1])
            with row1:
                st.text_input(
                    "clause text",
                    key=f"bc_{i}_text",
                    label_visibility="collapsed",
                    placeholder=f"Clause {i + 1} …",
                )
            with row2:
                st.selectbox(
                    "mode",
                    ["semantic", "keyword"],
                    key=f"bc_{i}_mode",
                    label_visibility="collapsed",
                )
            if i < n_bc - 1:
                if i in splits_set:
                    st.caption("— **Between groups** —")
                    st.selectbox(
                        f"Between groups: combine through Clause {i + 1} / Clause {i + 2}",
                        ["AND", "OR", "NOT"],
                        key=f"bc_between_{i}_op",
                    )
                else:
                    st.selectbox(
                        f"Within group: Clause {i + 1} with Clause {i + 2}",
                        ["AND", "OR", "NOT"],
                        key=f"bc_{i}_op",
                    )

        preview_clauses = [
            {
                "text": st.session_state.get(f"bc_{i}_text", ""),
                "mode": st.session_state.get(f"bc_{i}_mode", "semantic"),
            }
            for i in range(n_bc)
        ]
        preview_edges: list[str] = []
        for j in range(max(0, n_bc - 1)):
            if j in splits_set:
                preview_edges.append(st.session_state.get(f"bc_between_{j}_op", "AND"))
            else:
                preview_edges.append(st.session_state.get(f"bc_{j}_op", "AND"))
        preview_text = format_boolean_expression_preview(
            preview_clauses,
            sorted(splits_set),
            preview_edges,
        )
        st.markdown("**Boolean query**")
        st.text(preview_text if preview_text else "—")

        translation_md = format_boolean_expression_translation_md(
            preview_clauses,
            sorted(splits_set),
            preview_edges,
        )
        if translation_md:
            st.text("")
            st.markdown(
                "**Translation** — `(semantic)` / `(keyword)` · truncated text · "
                "boolean ops are colored: :green[**AND**] · :orange[**OR**] · :red[**NOT**]."
            )
            st.markdown(translation_md)

        col_cut, col_max = st.columns([1.2, 1.2])
        with col_cut:
            cutoff = st.number_input(
                "Min similarity",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("last_search_cutoff", 0.6)),
                step=0.01,
                format="%.2f",
                help="Semantic chunks below this are hidden; keyword chunks always shown.",
            )
        with col_max:
            max_str = st.text_input(
                "Max results",
                value=st.session_state.get("last_search_max_results_str", ""),
                placeholder="∞",
                help="Cap unique papers after filtering (empty = no limit).",
            )
            try:
                max_results = int(max_str) if max_str.strip() else None
            except ValueError:
                max_results = None

        search_btn = st.button("🔍 Search", type="primary", use_container_width=True, key="bool_search_btn")

        st.session_state["_tab1_cutoff_live"] = cutoff
        st.session_state["_tab1_max_results_live"] = max_results

        if search_btn:
            n_bc = int(st.session_state["bool_clause_count"])
            splits_run = _sanitize_bool_splits(
                n_bc, list(st.session_state.get("bool_group_splits") or [])
            )
            splits_set_run = set(splits_run)
            texts = [st.session_state.get(f"bc_{i}_text", "").strip() for i in range(n_bc)]
            if not any(texts):
                st.warning("Enter at least one non-empty clause.")
            elif n_bc > 1 and any(not texts[i] for i in range(n_bc)):
                st.warning(
                    "When using multiple clauses, fill **every** clause row (or remove extras)."
                )
            else:
                clauses = [
                    {"text": texts[i], "mode": st.session_state.get(f"bc_{i}_mode", "semantic")}
                    for i in range(n_bc)
                ]
                edge_ops: list[str] = []
                for j in range(max(0, n_bc - 1)):
                    if j in splits_set_run:
                        edge_ops.append(st.session_state.get(f"bc_between_{j}_op", "AND"))
                    else:
                        edge_ops.append(st.session_state.get(f"bc_{j}_op", "AND"))
                with st.spinner("Searching…"):
                    merged, _fps, label = boolean_retrieval_segmented(
                        clauses,
                        splits_run,
                        edge_ops,
                        embed_model,
                        DB_PATH,
                        n_results=500,
                    )
                st.session_state["search_results"] = merged
                st.session_state["last_search_query"] = label
                st.session_state["quick_chat_history"] = []
                st.session_state["search_generation"] = (
                    st.session_state.get("search_generation", 0) + 1
                )
                st.session_state["last_search_cutoff"] = cutoff
                st.session_state["last_search_max_results"] = max_results
                st.session_state["last_search_max_results_str"] = max_str.strip()
                st.rerun()

    _boolean_runner = (
        st.fragment()(_boolean_search_panel_fragment)
        if _streamlit_supports_fragment()
        else _boolean_search_panel_fragment
    )
    _boolean_runner()

    raw_results = st.session_state.get("search_results", [])
    cutoff = float(
        st.session_state.get(
            "_tab1_cutoff_live",
            st.session_state.get("last_search_cutoff", 0.6),
        )
    )
    max_results = st.session_state.get("_tab1_max_results_live")
    sorted_papers, papers_map = _compute_filtered_papers(raw_results, cutoff, max_results)

    hit_fps = {fp for fp, _ in sorted_papers}
    manual_extra = list(st.session_state.get("manual_extra_fps") or [])
    manual_only = [fp for fp in manual_extra if fp not in hit_fps]

    query_label = st.session_state.get("last_search_query") or ""

    if sorted_papers:
        pct = int(cutoff * 100)
        ql = query_label or "(no Boolean query label yet)"
        with st.expander(
            f"📂 Search results — **{len(sorted_papers)}** papers · "
            f"**Boolean query:** {ql} · "
            f"_Similarity filter: semantic ≥ {pct}% (keyword Clause rows exempt)._",
            expanded=False,
        ):
            gen = st.session_state.get("search_generation", 0)
            if st.session_state.get("_checkbox_gen_sync") != gen:
                for fp, _ in sorted_papers:
                    st.session_state[f"sel_{fp}"] = True
                st.session_state["_checkbox_gen_sync"] = gen

            for fp, hits in sorted_papers:
                best_score = max(h["score"] for h in hits)
                m0 = hits[0]["metadata"]
                score_pct = int(best_score * 100)
                color = _score_color(best_score)
                pdf_link = pdf_url(fp, PAPERS_DIR)

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

                abs_rec = load_abstract_record(fp, PAPERS_DIR)
                with st.expander("📋 Abstract", expanded=False):
                    if abs_rec:
                        st.caption(
                            f"status: **{abs_rec['status']}** · source: `{abs_rec['source']}` · "
                            f"{abs_rec['char_count']} chars · {abs_rec['word_count']} words"
                        )
                        if abs_rec.get("warnings"):
                            st.caption("⚠️ " + "; ".join(abs_rec["warnings"]))
                        st.write(abs_rec["abstract_text"])
                    else:
                        st.warning(
                            "No abstract JSON for this paper. From the app folder run:\n\n"
                            "`python extract_abstracts.py`"
                        )

                with st.expander(f"📑 Matching excerpts ({len(hits)})", expanded=False):
                    for hit in sorted(hits, key=lambda h: h["score"], reverse=True):
                        hm = hit["metadata"]
                        hp = int(hit["score"] * 100)
                        fname_hit = hm.get("file_name") or Path(hm["file_path"]).name
                        hit_badge = " · 🔑 keyword" if hit.get("match_type") == "keyword" else ""
                        st.markdown(
                            f"**`{fname_hit}`** · **Page {hm['page_num']}** — "
                            f":{_score_color(hit['score'])}[{hp}% similarity]{hit_badge}"
                        )
                        st.write(hit["text"])
                        st.markdown("---")
    elif raw_results:
        st.info(
            f"**Boolean query:** {query_label}\n\n"
            f"No papers above **{int(cutoff * 100)}%** similarity under the current filter "
            f"(semantic chunks only; **Clause** rows set to **keyword** ignore this cutoff)."
        )
    elif query_label:
        st.caption(
            f"Last **Boolean query:** {query_label} — adjust Clauses or cutoff."
        )

    if sorted_papers or manual_only:
        _selection_runner = (
            st.fragment()(_render_paper_selection_widgets)
            if _streamlit_supports_fragment()
            else _render_paper_selection_widgets
        )
        _selection_runner(sorted_papers, manual_only, papers_map, embed_model, gemini_client)


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
