"""Discover and download a small multimodal sample dataset into ./data/raw/showcase."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DownloadRecord:
    group: str
    query: str
    provider: str
    url: str
    target_path: str
    status: str
    detail: str = ""


GROUP_CONFIG = {
    "pdf": {
        "folder": "pdf",
        "extensions": [".pdf"],
        "queries": [
            "sample report filetype:pdf",
            "whitepaper example filetype:pdf",
            "guide example filetype:pdf",
        ],
    },
    "word": {
        "folder": "word",
        "extensions": [".docx"],
        "queries": [
            "sample proposal filetype:docx",
            "sample resume filetype:docx",
            "example contract filetype:docx",
        ],
    },
    "excel": {
        "folder": "excel",
        "extensions": [".xlsx"],
        "queries": [
            "sample budget filetype:xlsx",
            "sample workbook filetype:xlsx",
            "example spreadsheet filetype:xlsx",
        ],
    },
    "powerpoint": {
        "folder": "powerpoint",
        "extensions": [".pptx"],
        "queries": [
            "sample presentation filetype:pptx",
            "example slide deck filetype:pptx",
            "demo pitch deck filetype:pptx",
        ],
    },
    "images": {
        "folder": "images",
        "extensions": [".jpg", ".png", ".gif"],
        "queries": [
            "sample image jpg",
            "sample image png",
            "sample animated gif",
        ],
    },
    "videos": {
        "folder": "videos",
        "extensions": [".mp4"],
        "queries": [
            "sample video mp4",
            "demo clip mp4",
            "training sample mp4",
        ],
    },
    "audio": {
        "folder": "audio",
        "extensions": [".mp3"],
        "queries": [
            "sample audio mp3",
            "demo speech mp3",
            "podcast sample mp3",
        ],
    },
}


def provider_status() -> dict:
    return {
        "openai_web_search": bool(os.getenv("OPENAI_API_KEY")),
        "google_custom_search": bool(os.getenv("GOOGLE_SEARCH_API_KEY"))
        and bool(os.getenv("GOOGLE_SEARCH_ENGINE_ID")),
    }


def _openai_web_search(query: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    payload = {
        "model": os.getenv("OPENAI_SEARCH_MODEL", "gpt-5"),
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "input": query,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    urls: list[str] = []
    for output in data.get("output", []):
        if output.get("type") == "web_search_call":
            action = output.get("action", {})
            for source in action.get("sources", []):
                url = source.get("url")
                if isinstance(url, str):
                    urls.append(url)
        elif output.get("type") == "message":
            for content in output.get("content", []):
                for annotation in content.get("annotations", []):
                    if annotation.get("type") == "url_citation":
                        url = annotation.get("url")
                        if isinstance(url, str):
                            urls.append(url)
    return list(dict.fromkeys(urls))


def _google_custom_search(query: str, extensions: list[str]) -> list[str]:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if not api_key or not engine_id:
        return []

    search_type = "image" if set(extensions) & {".jpg", ".png", ".gif"} else None
    params = {
        "key": api_key,
        "cx": engine_id,
        "q": query,
        "num": "10",
    }
    if search_type:
        params["searchType"] = search_type

    url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    urls: list[str] = []
    for item in data.get("items", []):
        link = item.get("link")
        if isinstance(link, str):
            urls.append(link)
    return list(dict.fromkeys(urls))


def _candidate_urls(query: str, extensions: list[str], providers: list[str]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for provider in providers:
        if provider == "openai":
            for url in _openai_web_search(query):
                candidates.append((provider, url))
        elif provider == "google":
            for url in _google_custom_search(query, extensions):
                candidates.append((provider, url))
    return candidates


def _matches_extension(url: str, extensions: list[str]) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in extensions)


def _download(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target_path.write_bytes(response.read())


def create_sample_input_dataset(
    root: Path,
    max_per_group: int = 3,
    providers: list[str] | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    active_providers = providers or [name for name, enabled in provider_status().items() if enabled]
    normalized_providers = [
        "openai" if item == "openai_web_search" else "google" if item == "google_custom_search" else item
        for item in active_providers
    ]

    records: list[DownloadRecord] = []

    for group, config in GROUP_CONFIG.items():
        folder = root / config["folder"]
        folder.mkdir(parents=True, exist_ok=True)

        downloaded_count = 0
        for query in config["queries"]:
            for provider, url in _candidate_urls(query, config["extensions"], normalized_providers):
                if downloaded_count >= max_per_group:
                    break
                if not _matches_extension(url, config["extensions"]):
                    continue

                suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
                target = folder / f"{group}_{downloaded_count + 1:02d}{suffix}"
                try:
                    _download(url, target)
                    downloaded_count += 1
                    records.append(
                        DownloadRecord(
                            group=group,
                            query=query,
                            provider=provider,
                            url=url,
                            target_path=str(target),
                            status="downloaded",
                        )
                    )
                except Exception as exc:
                    records.append(
                        DownloadRecord(
                            group=group,
                            query=query,
                            provider=provider,
                            url=url,
                            target_path=str(target),
                            status="failed",
                            detail=str(exc),
                        )
                    )
            if downloaded_count >= max_per_group:
                break

    return {
        "root": str(root),
        "providers": normalized_providers,
        "status": provider_status(),
        "downloads": [asdict(item) for item in records],
    }
