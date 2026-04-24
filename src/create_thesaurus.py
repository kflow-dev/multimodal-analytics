"""Build a thesaurus-style inventory for local document collections."""

from __future__ import annotations

from pathlib import Path

from src.document_inventory import collect_inventory_from_root, summarize_inventory


def create_thesaurus(root: Path, recursive: bool = True) -> dict:
    """Return a thesaurus-style summary for all files under ``root``."""
    entries = collect_inventory_from_root(root, recursive=recursive)
    return summarize_inventory(entries)
