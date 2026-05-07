"""
rag_engine.py — Retrieval-Augmented Generation using Google Gemini (google-genai SDK).

Combines semantic search results from ChromaDB with Google Gemini
to generate grounded, cited answers over the paper collection.

Authentication: uses Application Default Credentials (ADC) via Vertex AI.
  - Set up with: gcloud auth application-default login
  - Quota project: gcloud auth application-default set-quota-project <project-id>
  - PDFs are uploaded to Google Cloud Storage (gs://) for size-unlimited file access.
  - Billing drawn from Google Cloud credits.
"""

from google import genai
from google.genai import types
from fastembed import TextEmbedding
from indexer import semantic_search, DB_PATH

# ── Configuration ─────────────────────────────────────────────────────────────

GCP_PROJECT        = "project-0afc802d-9b70-430c-9cc"
GCP_LOCATION       = "us-central1"
GCS_BUCKET         = "papers-rag-pdfs"
GEMINI_MODEL       = "gemini-2.5-flash"
MAX_CONTEXT_CHUNKS = 8

SYSTEM_INSTRUCTION = """You are a scientific literature assistant helping a researcher 
analyze papers in genomics, neuroscience, single-cell biology, and related fields.

Your role:
- Answer questions based on the provided paper excerpts
- Always cite sources using [Source N: Paper Title, p.X] notation
- Be precise and use appropriate scientific terminology
- If excerpts don't fully answer the question, say so clearly
- When synthesizing across multiple papers, explicitly compare and contrast
- Never fabricate citations or data not present in the excerpts"""


# ── Gemini client ─────────────────────────────────────────────────────────────

def get_gemini_client() -> genai.Client:
    """
    Initialize and return a Gemini client using Vertex AI + Application Default Credentials.
    PDFs are stored in GCS (gs://papers-rag-pdfs) for size-unlimited access.
    """
    return genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )


# ── Prompt building ───────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered source block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk["metadata"]
        parts.append(
            f"[Source {i}: {m['paper_title']}, p.{m['page_num']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_rag_message(query: str, chunks: list[dict]) -> str:
    """Combine retrieved context with the user query into a single message."""
    context = _build_context_block(chunks)
    return (
        f"Here are relevant excerpts from the paper collection:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Based on the excerpts above, please answer:\n{query}"
    )


# ── Streaming chat ────────────────────────────────────────────────────────────

def stream_rag_response(
    query: str,
    chat_history: list[dict],
    embedding_model: TextEmbedding,
    gemini_client: genai.Client,
    selected_file_paths: list[str] | None = None,
    n_chunks: int = MAX_CONTEXT_CHUNKS,
    db_path: str = DB_PATH,
    preloaded_chunks: list[dict] | None = None,
):
    """
    Generator yielding (text_chunk, sources) tuples.

    The very first yield contains (None, sources_list) so the caller can
    display source cards before streaming begins. Subsequent yields are
    (text: str, None).

    If `preloaded_chunks` is provided the internal semantic search is skipped
    and those chunks are used directly as context.  This is used by Quick Chat
    so that all papers found in the search are always represented, regardless
    of the MAX_CONTEXT_CHUNKS limit.
    """
    if preloaded_chunks is not None:
        chunks = preloaded_chunks
    else:
        chunks = semantic_search(
            query=query,
            embedding_model=embedding_model,
            db_path=db_path,
            n_results=n_chunks,
            file_paths=selected_file_paths,
        )

    if not chunks:
        yield "", []
        return

    # Emit sources immediately so UI can render them
    yield None, chunks

    # Build conversation history for multi-turn chat
    history = []
    for msg in chat_history:
        role = msg["role"]  # "user" or "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    chat_session = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        history=history,
    )

    rag_message = _build_rag_message(query, chunks)

    for piece in chat_session.send_message_stream(rag_message):
        if piece.text:
            yield piece.text, None


def get_direct_answer(
    query: str,
    embedding_model: TextEmbedding,
    gemini_client: genai.Client,
    selected_file_paths: list[str] | None = None,
    n_chunks: int = MAX_CONTEXT_CHUNKS,
    db_path: str = DB_PATH,
) -> tuple[str, list[dict]]:
    """
    Non-streaming single-turn answer. Returns (answer_text, sources).
    Used by the Search tab.
    """
    chunks = semantic_search(
        query=query,
        embedding_model=embedding_model,
        db_path=db_path,
        n_results=n_chunks,
        file_paths=selected_file_paths,
    )

    if not chunks:
        return "No relevant content found for this query.", []

    rag_message = _build_rag_message(query, chunks)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=rag_message,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text, chunks


# ── Deep Chat: full PDF sent directly to Gemini ───────────────────────────────

PDF_SYSTEM_INSTRUCTION = """You are a scientific literature assistant with direct access 
to the full content of one or more research papers provided as PDFs.

Your role:
- Answer questions based on the complete paper content you have been given
- Cite specific sections, figures, or page numbers when relevant
- Be precise and use appropriate scientific terminology
- Compare and contrast across papers when multiple are provided
- If a question cannot be answered from the papers, say so clearly
- Never fabricate data or citations not present in the provided papers"""


def upload_pdfs_to_gcs(
    pdf_paths: list[str],
    progress_callback=None,
) -> dict[str, str | None]:
    """
    Upload PDFs to Google Cloud Storage (gs://papers-rag-pdfs/).

    Uses Application Default Credentials — no API key needed.
    Files persist until deleted (no 48 h expiry). Vertex AI reads gs:// URIs directly.

    Args:
        pdf_paths:         Local paths to PDF files.
        progress_callback: Optional callable(current, total, filename, success, error).

    Returns:
        dict mapping each local path to its gs:// URI (or None on failure).
    """
    from pathlib import Path as _Path
    from google.cloud import storage as _gcs

    gcs = _gcs.Client(project=GCP_PROJECT)
    bucket = gcs.bucket(GCS_BUCKET)
    results: dict[str, str | None] = {}
    total = len(pdf_paths)

    for i, path in enumerate(pdf_paths):
        name = _Path(path).name
        blob = bucket.blob(f"selected/{name}")
        gs_uri = f"gs://{GCS_BUCKET}/selected/{name}"
        try:
            # Use a long timeout (10 min) to handle large PDFs over slow connections.
            # GCS automatically uses resumable upload for files > 8 MB.
            file_size_mb = _Path(path).stat().st_size / (1024 * 1024)
            timeout = max(300, int(file_size_mb * 10))  # ~10s per MB, min 5 min
            blob.upload_from_filename(
                path,
                content_type="application/pdf",
                timeout=timeout,
            )
            results[path] = gs_uri
            if progress_callback:
                progress_callback(i + 1, total, name, True, "")
        except Exception as exc:
            # Verify the blob actually made it despite the exception (resumable uploads
            # sometimes complete on the server even if the client gets a network error).
            try:
                if blob.exists():
                    results[path] = gs_uri
                    if progress_callback:
                        progress_callback(i + 1, total, name, True, "")
                    continue
            except Exception:
                pass
            results[path] = None
            if progress_callback:
                progress_callback(i + 1, total, name, False, str(exc))
    return results


def stream_pdf_chat(
    query: str,
    pdf_paths: list[str],
    chat_history: list[dict],
    gemini_client: genai.Client,
    uploaded_files: dict | None = None,
) -> str:
    """
    Generator that streams Gemini responses for a direct full-PDF chat session.

    When `uploaded_files` is provided (from upload_pdfs_to_gemini), PDFs are
    referenced by their Gemini File URI — no inline bytes, no size limits.
    Without it, falls back to inline bytes (20 MB request cap applies).

    Args:
        query:          Current user question.
        pdf_paths:      List of absolute paths to PDF files to include.
        chat_history:   Prior turns as [{role, content}].
        gemini_client:  Initialized Gemini client (Vertex AI).
        uploaded_files: Optional dict {path: types.File} from upload_pdfs_to_gemini.

    Yields:
        str: Text chunks of the streaming response.
    """
    pdf_parts = []
    for path in pdf_paths:
        if uploaded_files and uploaded_files.get(path) is not None:
            gs_uri = uploaded_files[path]
            pdf_parts.append(
                types.Part.from_uri(file_uri=gs_uri, mime_type="application/pdf")
            )
        else:
            try:
                data = open(path, "rb").read()
                pdf_parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
            except Exception as exc:
                yield f"[Warning: could not read {path}: {exc}]\n"

    if not pdf_parts:
        yield "No PDF files could be loaded. Please check the file paths."
        return

    # Rebuild full conversation history for Gemini
    history = []
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user" and i == 0:
            parts = pdf_parts + [types.Part(text=msg["content"])]
            history.append(types.Content(role="user", parts=parts))
        else:
            history.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])],
            ))

    chat_session = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=PDF_SYSTEM_INSTRUCTION),
        history=history,
    )

    if not chat_history:
        current_content = pdf_parts + [types.Part(text=query)]
    else:
        current_content = query

    for piece in chat_session.send_message_stream(current_content):
        if piece.text:
            yield piece.text
