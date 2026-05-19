"""
pdf_server.py — Lightweight HTTP file server for serving PDFs.

Starts a background thread on port 8502 that serves the papers directory
as static files. This allows browser hyperlinks like:
    http://localhost:8502/130_kania.../paper.pdf
to open PDFs in a new browser tab (browsers block file:// links from
localhost web pages for security reasons).
"""

import http.server
import os
import threading
from pathlib import Path
from urllib.parse import quote

def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable, falling back to ``default``."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


PDF_SERVER_PORT = _env_int("PDF_SERVER_PORT", 8502)
_served_directory = os.getcwd()
_server_started = False


def _make_handler(directory: str):
    """Return a SimpleHTTPRequestHandler class serving the current PDF root."""
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=_served_directory, **kwargs)

        def log_message(self, format, *args):
            """Silence default ``GET …`` access lines on stderr."""
            pass  # suppress access logs in terminal

    return _Handler


def start_pdf_server(papers_dir: str, port: int = PDF_SERVER_PORT) -> bool:
    """
    Start a background HTTP server serving `papers_dir` on `port`.

    Returns True if started successfully, False if port already in use.
    Safe to call multiple times — only one server will start.
    """
    global _served_directory, _server_started
    _served_directory = str(Path(papers_dir).resolve())
    if _server_started:
        return False
    try:
        handler = _make_handler(papers_dir)
        server = http.server.HTTPServer(("127.0.0.1", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _server_started = True
        return True
    except OSError:
        # Port already in use — server already running
        return False


def pdf_url(file_path: str, papers_dir: str, port: int = PDF_SERVER_PORT) -> str:
    """
    Convert an absolute PDF file path to a localhost HTTP URL.

    Example:
        /home/mneira/MAURICIO/papers/130_kania/paper.pdf
        → http://localhost:8502/130_kania/paper.pdf
    """
    try:
        rel = Path(file_path).relative_to(papers_dir)
        # URL-encode path segments to handle spaces and special characters
        encoded = "/".join(quote(part) for part in rel.parts)
        return f"http://localhost:{port}/{encoded}"
    except ValueError:
        # file_path is not under papers_dir
        return ""
