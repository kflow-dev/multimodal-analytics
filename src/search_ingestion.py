"""Modular search and ingestion workflow reusable from CLI and web UI."""

from __future__ import annotations

import asyncio
import csv
import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".xhtml"}
PDF_EXTENSIONS = {".pdf"}

DOCUMENT_TYPE_EXTENSIONS = {
    "pdf": PDF_EXTENSIONS,
    "image": IMAGE_EXTENSIONS,
    "office": OFFICE_EXTENSIONS,
    "word": {".docx"},
    "excel": {".xlsx"},
    "powerpoint": {".pptx"},
    "text": TEXT_EXTENSIONS,
    "spreadsheet": {".xls", ".xlsx"},
    "presentation": {".ppt", ".pptx"},
    "document": PDF_EXTENSIONS | OFFICE_EXTENSIONS | TEXT_EXTENSIONS,
}


@dataclass
class SearchCandidate:
    source_type: str
    reference: str
    extension: str = ""
    document_type: str = "unknown"
    matched_keywords: list[str] = field(default_factory=list)
    local_path: str | None = None
    downloaded: bool = False


@dataclass
class SearchIngestionRequest:
    keywords: list[str]
    document_types: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    input_location: str | None = None
    url_subdomain_pattern: str | None = None
    output_location: str | None = None
    parse_method: str = "auto"
    workers: int = 1
    recursive: bool = True
    dry_run: bool = False


@dataclass
class SearchIngestionResult:
    request: dict
    candidates: list[dict] = field(default_factory=list)
    ingested_files: list[str] = field(default_factory=list)
    downloaded_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    output_location: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_csv(values: str | None) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def build_request(
    keywords: str | list[str],
    document_types: str | list[str] | None = None,
    extensions: str | list[str] | None = None,
    input_location: str | None = None,
    url_subdomain_pattern: str | None = None,
    output_location: str | None = None,
    parse_method: str = "auto",
    workers: int = 1,
    recursive: bool = True,
    dry_run: bool = False,
) -> SearchIngestionRequest:
    keyword_values = keywords if isinstance(keywords, list) else _normalize_csv(keywords)
    doc_type_values = (
        document_types
        if isinstance(document_types, list)
        else _normalize_csv(document_types)
    )
    extension_values = (
        extensions if isinstance(extensions, list) else _normalize_csv(extensions)
    )

    normalized_extensions = []
    for ext in extension_values:
        normalized_extensions.append(ext if ext.startswith(".") else f".{ext}")

    return SearchIngestionRequest(
        keywords=keyword_values,
        document_types=[item.lower() for item in doc_type_values],
        extensions=[item.lower() for item in normalized_extensions],
        input_location=input_location,
        url_subdomain_pattern=url_subdomain_pattern,
        output_location=output_location,
        parse_method=parse_method,
        workers=workers,
        recursive=recursive,
        dry_run=dry_run,
    )


def _get_document_type(extension: str) -> str:
    extension = extension.lower()
    preferred_order = [
        "word",
        "excel",
        "powerpoint",
        "pdf",
        "image",
        "text",
        "spreadsheet",
        "presentation",
        "office",
        "document",
    ]
    for name in preferred_order:
        extensions = DOCUMENT_TYPE_EXTENSIONS.get(name, set())
        if extension in extensions:
            return name
    return "unknown"


def _allowed_extensions(request: SearchIngestionRequest) -> set[str]:
    allowed = set(request.extensions)
    for doc_type in request.document_types:
        allowed |= DOCUMENT_TYPE_EXTENSIONS.get(doc_type, set())
    return {item.lower() for item in allowed}


def _matches_keywords(reference: str, keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    reference_lower = reference.lower()
    return [keyword for keyword in keywords if keyword.lower() in reference_lower]


def _matches_subdomain(url: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    hostname = urlparse(url).hostname or ""
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(hostname, pattern)
    return pattern in hostname


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _read_url_manifest(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list"}:
        return [line.strip() for line in path.read_text().splitlines() if _is_url(line.strip())]
    if suffix == ".json":
        content = json.loads(path.read_text())
        if isinstance(content, str):
            return [content] if _is_url(content) else []
        if isinstance(content, list):
            urls = []
            for item in content:
                if isinstance(item, str) and _is_url(item):
                    urls.append(item)
                elif isinstance(item, dict):
                    for key in ("url", "href", "link"):
                        value = item.get(key)
                        if isinstance(value, str) and _is_url(value):
                            urls.append(value)
                            break
            return urls
        return []
    if suffix == ".csv":
        urls: list[str] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in ("url", "href", "link"):
                    value = row.get(key)
                    if value and _is_url(value):
                        urls.append(value)
                        break
        return urls
    return []


def discover_candidates(request: SearchIngestionRequest) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    allowed_extensions = _allowed_extensions(request)
    base_input = request.input_location or "data/raw"
    input_path = Path(base_input)

    if _is_url(base_input):
        ext = Path(urlparse(base_input).path).suffix.lower()
        candidate = SearchCandidate(
            source_type="url",
            reference=base_input,
            extension=ext,
            document_type=_get_document_type(ext),
            matched_keywords=_matches_keywords(base_input, request.keywords),
        )
        return _filter_candidates([candidate], request, allowed_extensions)

    if input_path.is_file():
        manifest_urls = _read_url_manifest(input_path)
        if manifest_urls:
            for url in manifest_urls:
                ext = Path(urlparse(url).path).suffix.lower()
                candidates.append(
                    SearchCandidate(
                        source_type="url",
                        reference=url,
                        extension=ext,
                        document_type=_get_document_type(ext),
                        matched_keywords=_matches_keywords(url, request.keywords),
                    )
                )
        else:
            ext = input_path.suffix.lower()
            candidates.append(
                SearchCandidate(
                    source_type="local_file",
                    reference=str(input_path),
                    local_path=str(input_path),
                    extension=ext,
                    document_type=_get_document_type(ext),
                    matched_keywords=_matches_keywords(str(input_path), request.keywords),
                )
            )
        return _filter_candidates(candidates, request, allowed_extensions)

    if input_path.is_dir():
        pattern = "**/*" if request.recursive else "*"
        for candidate_path in input_path.glob(pattern):
            if not candidate_path.is_file():
                continue
            ext = candidate_path.suffix.lower()
            candidates.append(
                SearchCandidate(
                    source_type="local_file",
                    reference=str(candidate_path),
                    local_path=str(candidate_path),
                    extension=ext,
                    document_type=_get_document_type(ext),
                    matched_keywords=_matches_keywords(str(candidate_path), request.keywords),
                )
            )

    return _filter_candidates(candidates, request, allowed_extensions)


def _filter_candidates(
    candidates: Iterable[SearchCandidate],
    request: SearchIngestionRequest,
    allowed_extensions: set[str],
) -> list[SearchCandidate]:
    filtered: list[SearchCandidate] = []
    for candidate in candidates:
        if allowed_extensions and candidate.extension not in allowed_extensions:
            continue
        if request.keywords and not candidate.matched_keywords:
            continue
        if candidate.source_type == "url" and not _matches_subdomain(
            candidate.reference, request.url_subdomain_pattern
        ):
            continue
        filtered.append(candidate)
    return filtered


def _download_target_path(url: str, output_dir: Path) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if not suffix:
        suffix = ".bin"
    stem = Path(parsed.path).stem or "download"
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return output_dir / f"{stem}_{digest}{suffix}"


def download_url_candidates(
    candidates: list[SearchCandidate],
    output_dir: Path,
) -> tuple[list[SearchCandidate], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []

    for candidate in candidates:
        if candidate.source_type != "url":
            continue
        target_path = _download_target_path(candidate.reference, output_dir)
        with urlopen(candidate.reference) as response:
            target_path.write_bytes(response.read())
        candidate.local_path = str(target_path)
        candidate.downloaded = True
        downloaded_files.append(str(target_path))

    return candidates, downloaded_files


def _default_rag_factory():
    from src.runtime import build_rag

    return build_rag()


async def execute_search_ingestion(
    request: SearchIngestionRequest,
    rag_factory: Callable | None = None,
) -> SearchIngestionResult:
    output_location = request.output_location or os.getenv("OUTPUT_DIR", "./output")
    result = SearchIngestionResult(
        request=asdict(request),
        output_location=output_location,
    )

    candidates = discover_candidates(request)
    result.candidates = [asdict(candidate) for candidate in candidates]

    if request.dry_run or not candidates:
        return result

    output_dir = Path(output_location)
    try:
        candidates, downloaded_files = download_url_candidates(
            candidates, output_dir / "downloads"
        )
        result.downloaded_files = downloaded_files
    except Exception as exc:
        result.errors.append(f"download_failed: {exc}")
        return result

    local_paths = [candidate.local_path for candidate in candidates if candidate.local_path]
    if not local_paths:
        return result

    rag_builder = rag_factory or _default_rag_factory
    rag = rag_builder()
    try:
        ingestion_result = await rag.process_documents_with_rag_batch(
            file_paths=local_paths,
            output_dir=output_location,
            parse_method=request.parse_method,
            max_workers=request.workers,
            recursive=request.recursive,
        )
        rag_results = ingestion_result.get("rag_results", {})
        for file_path, file_result in rag_results.items():
            if file_result.get("processed"):
                result.ingested_files.append(file_path)
            else:
                result.errors.append(f"{file_path}: {file_result.get('error', 'failed')}")
    finally:
        await rag._async_close()

    return result


def execute_search_ingestion_sync(
    request: SearchIngestionRequest,
    rag_factory: Callable | None = None,
) -> SearchIngestionResult:
    return asyncio.run(execute_search_ingestion(request, rag_factory=rag_factory))
