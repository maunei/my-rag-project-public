"""
papers_paths.py — PDF root path and discovery only (stdlib).

Keeps CLI tools like extract_abstracts.py free of chromadb / sqlite imports.

``PAPERS_DIR`` comes from the last selected app library, falling back to
``.env`` via ``papers_rag_config``.
"""

import os
from pathlib import Path

from papers_rag_config import INDEX_DIR_NAME, PAPERS_DIR


def get_all_pdfs(papers_dir: str = PAPERS_DIR) -> list[str]:
    """Recursively collect source PDF paths, excluding internal app data."""
    root = Path(papers_dir)
    pdfs: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != INDEX_DIR_NAME]
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdfs.append(str(Path(dirpath) / filename))
    return sorted(pdfs)
