"""
papers_paths.py — PDF root path and discovery only (stdlib).

Keeps CLI tools like extract_abstracts.py free of chromadb / sqlite imports.

``PAPERS_DIR`` comes from ``.env`` via ``papers_rag_config`` (mandatory ``PAPERS_DIR`` variable).
"""

from pathlib import Path

from papers_rag_config import PAPERS_DIR


def get_all_pdfs(papers_dir: str = PAPERS_DIR) -> list[str]:
    """Recursively collect all PDF paths under papers_dir, sorted."""
    return sorted(str(p) for p in Path(papers_dir).rglob("*.pdf"))
