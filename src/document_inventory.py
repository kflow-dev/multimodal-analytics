"""Inventory and thesaurus helpers for notebook-driven document exploration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class InventoryEntry:
    path: str
    name: str
    extension: str
    group: str
    size_bytes: int


GROUP_RULES = {
    "pdf": {".pdf"},
    "word": {".docx"},
    "excel": {".xlsx"},
    "powerpoint": {".pptx"},
    "image": {".jpg", ".png", ".gif"},
    "video": {".mp4"},
    "audio": {".mp3"},
    "markdown": {".md"},
    "code": {".txt", ".json", ".yaml", ".yml"},
}

SHOWCASE_TARGETS = {
    "pdf": [".pdf"],
    "word": [".docx"],
    "excel": [".xlsx"],
    "powerpoint": [".pptx"],
    "image": [".jpg", ".png", ".gif"],
    "video": [".mp4"],
    "audio": [".mp3"],
    "markdown": [".md"],
    "code": [".txt", ".json", ".yaml"],
    "text_chunks_or_urls": [".txt", "url-manifest"],
}


def classify_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    for group, extensions in GROUP_RULES.items():
        if suffix in extensions:
            return group
    return "other"


def collect_inventory(paths: Iterable[Path]) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for path in paths:
        if not path.is_file():
            continue
        entries.append(
            InventoryEntry(
                path=str(path),
                name=path.name,
                extension=path.suffix.lower(),
                group=classify_extension(path),
                size_bytes=path.stat().st_size,
            )
        )
    return sorted(entries, key=lambda entry: (entry.group, entry.name))


def collect_inventory_from_root(root: Path, recursive: bool = True) -> list[InventoryEntry]:
    if recursive:
        paths = root.rglob("*")
    else:
        paths = root.glob("*")
    return collect_inventory(paths)


def build_thesaurus(entries: Iterable[InventoryEntry]) -> dict[str, list[dict]]:
    thesaurus: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        thesaurus[entry.group].append(asdict(entry))
    return dict(sorted(thesaurus.items()))


def summarize_inventory(entries: Iterable[InventoryEntry]) -> dict:
    entry_list = list(entries)
    thesaurus = build_thesaurus(entry_list)
    return {
        "total_files": len(entry_list),
        "total_size_bytes": sum(item.size_bytes for item in entry_list),
        "groups": {group: len(items) for group, items in thesaurus.items()},
        "files": [asdict(item) for item in entry_list],
        "thesaurus": thesaurus,
    }


def showcase_coverage(entries: Iterable[InventoryEntry]) -> dict[str, dict]:
    by_extension: dict[str, int] = defaultdict(int)
    for entry in entries:
        by_extension[entry.extension] += 1

    coverage: dict[str, dict] = {}
    for label, targets in SHOWCASE_TARGETS.items():
        available = {target: by_extension.get(target, 0) for target in targets if target.startswith(".")}
        coverage[label] = {
            "required": 3,
            "targets": targets,
            "available": available,
            "meets_requirement": sum(available.values()) >= 3 if available else False,
        }
    return coverage
