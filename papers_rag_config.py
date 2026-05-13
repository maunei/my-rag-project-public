"""
papers_rag_config.py — Load ``.env`` and required directory roots (no literals in callers).

Raises ``RuntimeError`` on startup if mandatory variables are unset or unusable paths.
"""

from __future__ import annotations

import os
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent


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


def _require_writable_dirs(env_name: str) -> Path:
    path = _resolve_path(_require_env(env_name))
    path.mkdir(parents=True, exist_ok=True)
    return path


# PDF corpus root (must already exist — never auto-create user's library path).
papers_path = _resolve_path(_require_env("PAPERS_DIR"))
if not papers_path.is_dir():
    raise RuntimeError(
        f"PAPERS_DIR is not an existing directory: {papers_path} "
        f"(configure PAPERS_DIR in {_APP_ROOT / '.env'})"
    )
PAPERS_DIR = str(papers_path)

# Chroma persistence directory (Chromadb PersistentClient handles contents).
chromadb_path = _resolve_path(_require_env("CHROMA_DB_PATH"))
chromadb_path.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH = str(chromadb_path)

ABSTRACT_META_ROOT = _require_writable_dirs("ABSTRACT_META_ROOT")

EXPORTED_PROMPTS_DIR = _require_writable_dirs("EXPORTED_PROMPTS_DIR")

# Base dir for staged Deep Chat symlinks/copies (timestamp subfolders at runtime).
SELECTED_PDFS_BASE = _require_writable_dirs("SELECTED_PDFS_DIR")
