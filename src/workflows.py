"""Reusable workflow functions for RAG Anything pipelines."""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path


async def run_document_ingestion(
    rag: Any,
    file_paths: List[str],
    output_dir: Optional[str] = None,
    parse_method: str = "auto",
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Ingest documents into the RAG pipeline.

    Args:
        rag: AIFluent instance
        file_paths: List of file paths to process
        output_dir: Output directory for parsed content
        parse_method: Parsing method (auto, ocr, txt)
        max_workers: Concurrent processing workers

    Returns:
        Dict with ingestion results
    """
    start = time.time()
    results = {"files": [], "errors": [], "duration": 0}

    if len(file_paths) == 1:
        fp = file_paths[0]
        if Path(fp).is_dir():
            await rag.process_folder_complete(
                fp, output_dir=output_dir, parse_method=parse_method, max_workers=max_workers
            )
        else:
            await rag.process_document_complete(
                fp, output_dir=output_dir, parse_method=parse_method
            )
        results["files"].append(fp)
    else:
        result = await rag.process_documents_with_rag_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            max_workers=max_workers,
        )
        if isinstance(result, dict):
            results["files"] = result.get("successful_rag_files", [])
            results["errors"] = result.get("failed_files", [])
            results["total_files"] = len(result.get("parse_result", {}).get("successful_files", []))

    results["duration"] = time.time() - start
    return results


async def run_multimodal_query(
    rag: Any,
    query: str,
    mode: str = "mix",
    **kwargs,
) -> str:
    """Execute a multimodal query.

    Args:
        rag: AIFluent instance
        query: Query text
        mode: Query mode (local, global, hybrid, naive, mix, bypass)
        **kwargs: Additional query parameters

    Returns:
        Query response string
    """
    return await rag.aquery(query, mode=mode, **kwargs)


async def run_batch_processing(
    rag: Any,
    file_paths: List[str],
    output_dir: Optional[str] = None,
    parse_method: str = "auto",
    recursive: bool = True,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Process documents in batch mode.

    Args:
        rag: AIFluent instance
        file_paths: List of file paths or directories
        output_dir: Output directory
        parse_method: Parsing method
        recursive: Recurse into subfolders
        max_workers: Concurrent workers

    Returns:
        Dict with batch results
    """
    if output_dir is None:
        output_dir = rag.config.parser_output_dir

    return await rag.process_documents_with_rag_batch(
        file_paths=file_paths,
        output_dir=output_dir,
        parse_method=parse_method,
        recursive=recursive,
        max_workers=max_workers,
    )


def format_rag_results(results: Any) -> str:
    """Format RAG retrieval results for display."""
    if hasattr(results, "format_results"):
        return results.format_results(results.get("per_query_metrics", []))
    return str(results)
