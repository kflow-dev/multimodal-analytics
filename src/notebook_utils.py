"""Shared notebook utilities for RAG Anything pipelines."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
CHECKPOINT_DATA = DATA_DIR / "checkpoints"
STORAGE_DIR = PROJECT_ROOT / "rag_storage"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_project_root_on_path() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(name: str) -> Path:
    """Get a data file path under raw/."""
    return RAW_DATA / name
