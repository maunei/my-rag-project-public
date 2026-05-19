"""
papers_rag_config.py — Load app defaults and derive per-library working folders.

``PAPERS_DIR`` comes from the last selected app library when available, falling
back to ``.env``. All local data folders are derived under
``PAPERS_DIR/papers-rag_index``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
_LIBRARY_STATE_PATH = _APP_ROOT / ".papers_rag_state.json"
INDEX_DIR_NAME = "papers-rag_index"


def _fallback_load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from KEY=VALUE lines when python-dotenv is unavailable."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        val = value.strip()
        if (len(val) >= 2 and val[0] == val[-1] == '"') or (
            len(val) >= 2 and val[0] == val[-1] == "'"
        ):
            val = val[1:-1]
        if not key:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = val


try:
    from dotenv import load_dotenv

    load_dotenv(_APP_ROOT / ".env")
except ImportError:
    _fallback_load_dotenv(_APP_ROOT / ".env")


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set or empty. Add it to {_APP_ROOT / '.env'} "
            "(see .env.example)."
        )
    return value


def _resolve_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw.strip()))).resolve()


def _state_papers_dir() -> Path | None:
    try:
        with _LIBRARY_STATE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    raw = str(data.get("papers_dir") or "").strip()
    if not raw:
        return None
    path = _resolve_path(raw)
    return path if path.is_dir() else None


# PDF corpus root (must already exist — never auto-create user's library path).
papers_path = _state_papers_dir() or _resolve_path(_require_env("PAPERS_DIR"))
if not papers_path.is_dir():
    raise RuntimeError(
        f"PAPERS_DIR is not an existing directory: {papers_path} "
        f"(configure PAPERS_DIR in {_APP_ROOT / '.env'})"
    )
PAPERS_DIR = str(papers_path)

INDEX_ROOT = papers_path / INDEX_DIR_NAME
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# Derived per-library directories.
chromadb_path = INDEX_ROOT / "chroma_db"
chromadb_path.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH = str(chromadb_path)

ABSTRACT_META_ROOT = INDEX_ROOT / "abstract_meta"
ABSTRACT_META_ROOT.mkdir(parents=True, exist_ok=True)

EXPORTED_PROMPTS_DIR = INDEX_ROOT / "exported_prompts"
EXPORTED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Base dir for staged Deep Chat symlinks/copies (timestamp subfolders at runtime).
SELECTED_PDFS_BASE = INDEX_ROOT / "selected_pdfs"
SELECTED_PDFS_BASE.mkdir(parents=True, exist_ok=True)

DATABASE_HEALTH_REPORTS_DIR = INDEX_ROOT / "databases_health_reports"
DATABASE_HEALTH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
