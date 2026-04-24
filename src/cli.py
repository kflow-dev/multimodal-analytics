"""CLI entry points for RAG Anything."""

import asyncio
import json
import click

from src.search_ingestion import build_request, execute_search_ingestion
from src.showcase_rag import (
    ShowcaseTechniques,
    index_showcase_dataset_sync,
    query_showcase_thesaurus_sync,
    summarize_showcase_sync,
)
from src.web_ui import serve_web_ui


@click.group()
def cli():
    """RAG Anything - Multimodal Document Processing and Retrieval Pipeline."""
    pass


@cli.command()
@click.argument("file_paths", nargs=-1)
@click.option("--output", "-o", default=None, help="Output directory for parsed content")
@click.option("--method", "-m", default="auto", help="Parsing method (auto, ocr, txt)")
@click.option("--parser", "-p", default="mineru", help="Parser to use (mineru, docling, paddleocr)")
@click.option("--workers", "-w", default=1, help="Max concurrent workers")
@click.option("--recursive", "-r", is_flag=True, help="Process folders recursively")
@click.option("--stats", is_flag=True, help="Display content statistics")
def ingest(file_paths, output, method, parser, workers, recursive, stats):
    """Ingest documents into the RAG pipeline."""

    async def _ingest():
        from src.runtime import build_rag

        rag = build_rag()
        previous_parser = rag.config.parser
        results = {"files": [], "errors": [], "duration": 0}
        import time
        start = time.time()

        try:
            if parser:
                rag.config.parser = parser
            for fp in file_paths:
                try:
                    from pathlib import Path

                    if Path(fp).is_dir():
                        await rag.process_folder_complete(
                            fp,
                            output_dir=output,
                            parse_method=method,
                            max_workers=workers,
                            display_stats=stats,
                            recursive=recursive,
                        )
                    else:
                        await rag.process_document_complete(
                            fp,
                            output_dir=output,
                            parse_method=method,
                            display_stats=stats,
                        )
                    results["files"].append(fp)
                except Exception as e:
                    results["errors"].append({"file": fp, "error": str(e)})

            results["duration"] = time.time() - start
            return results
        finally:
            rag.config.parser = previous_parser
            await rag._async_close()

    result = asyncio.run(_ingest())
    click.echo(f"Ingested {len(result['files'])} files, {len(result['errors'])} errors")
    if result["errors"]:
        for e in result["errors"]:
            click.echo(f"  Error: {e['file']} -> {e['error']}")


@cli.command()
@click.argument("query")
@click.option("--mode", "-m", default="mix", help="Query mode (local, global, hybrid, naive, mix, bypass)")
def query(query, mode):
    """Execute a query against the RAG pipeline."""

    async def _query():
        from src.runtime import build_rag

        rag = build_rag()
        try:
            return await rag.aquery(query, mode=mode)
        finally:
            await rag._async_close()

    result = asyncio.run(_query())
    click.echo(result)


@cli.command(name="evaluate")
@click.argument("test_file")
@click.option("--k-values", default="1,5,10", help="Comma-separated k values")
@click.option("--output", "-o", default=None, help="Output JSON file for results")
def evaluate_cmd(test_file, k_values, output):
    """Evaluate RAG pipeline on test queries."""
    with open(test_file) as f:
        test_cases = json.load(f)

    test_queries = [(tc["query"], set(tc.get("relevant_ids", []))) for tc in test_cases]
    k_values_list = [int(k) for k in k_values.split(",")]

    from src.runtime import build_rag

    rag = build_rag()
    results = asyncio.run(rag.evaluate(test_queries, k_values=k_values_list))
    rag.print_evaluation_comparison(results)

    if output:
        rag.save_evaluation_results(output)
    asyncio.run(rag._async_close())


@cli.command()
@click.argument("file_paths", nargs=-1)
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--workers", "-w", default=1, help="Max concurrent workers")
@click.option("--recursive", "-r", is_flag=True, help="Process recursively")
def batch(file_paths, output, workers, recursive):
    """Process multiple documents in batch mode."""

    async def _batch():
        from src.runtime import build_rag

        rag = build_rag()
        try:
            return await rag.process_documents_with_rag_batch(
                file_paths=file_paths,
                output_dir=output,
                max_workers=workers,
                recursive=recursive,
            )
        finally:
            await rag._async_close()

    result = asyncio.run(_batch())
    click.echo(f"Batch complete: {result}")


@cli.command()
@click.option("--check", is_flag=True, help="Check parser installation")
def status(check):
    """Show RAG Anything status and configuration."""
    from aifluent import AIFluentConfig

    config = AIFluentConfig()

    if check:
        from aifluent.parser import SUPPORTED_PARSERS, get_parser
        for parser_name in SUPPORTED_PARSERS:
            installed = get_parser(parser_name).check_installation()
            status_str = "OK" if installed else "MISSING"
            click.echo(f"  {parser_name}: {status_str}")
    else:
        click.echo("RAG Anything Configuration:")
        click.echo(f"  Parser: {config.parser}")
        click.echo(f"  Parse method: {config.parse_method}")
        click.echo(f"  Working dir: {config.working_dir}")
        click.echo(f"  Multimodal - Image: {config.enable_image_processing}, "
                    f"Table: {config.enable_table_processing}, "
                    f"Equation: {config.enable_equation_processing}")


@cli.command(name="search-ingest")
@click.option("--keywords", required=True, help="Comma-separated keywords to match")
@click.option("--document-types", default="", help="Optional comma-separated document types")
@click.option("--extensions", default="", help="Optional comma-separated file extensions")
@click.option("--input-location", default=None, help="Optional file, folder, URL, or URL manifest")
@click.option("--url-subdomain-pattern", default=None, help="Optional hostname filter")
@click.option("--output-location", default=None, help="Optional parser/download output location")
@click.option("--parse-method", default="auto", help="Parsing method")
@click.option("--workers", default=1, type=int, help="Concurrent ingestion workers")
@click.option("--recursive/--no-recursive", default=True, help="Search local folders recursively")
@click.option("--dry-run", is_flag=True, help="Search only, do not ingest")
def search_ingest_cmd(
    keywords,
    document_types,
    extensions,
    input_location,
    url_subdomain_pattern,
    output_location,
    parse_method,
    workers,
    recursive,
    dry_run,
):
    """Search local files/URL manifests and ingest matching documents."""
    request = build_request(
        keywords=keywords,
        document_types=document_types,
        extensions=extensions,
        input_location=input_location,
        url_subdomain_pattern=url_subdomain_pattern,
        output_location=output_location,
        parse_method=parse_method,
        workers=workers,
        recursive=recursive,
        dry_run=dry_run,
    )
    result = asyncio.run(execute_search_ingestion(request))
    click.echo(json.dumps(result.to_dict(), indent=2))


@cli.command(name="serve-ui")
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
def serve_ui_cmd(host, port):
    """Serve the search and ingestion web UI."""
    serve_web_ui(host=host, port=port)


@cli.command(name="showcase-rag")
@click.option("--input-location", default="./data/raw/showcase", help="Showcase dataset root")
@click.option("--output-location", default="./output/showcase", help="Parser output location")
@click.option("--parser", default="mineru", help="Parser selection")
@click.option("--parse-method", default="auto", help="Parsing method")
@click.option("--query-mode", default="mix", help="Query mode")
@click.option("--response-type", default="Multiple Paragraphs", help="Response type")
@click.option("--query-optimization-strategy", default="all", help="Query optimization strategy")
@click.option("--context-window", default=1, type=int, help="Context window")
@click.option("--top-k", default=10, type=int, help="Query top-k")
@click.option("--enable-vlm/--disable-vlm", default=False, help="Enable VLM-enhanced querying")
@click.option("--index-only", is_flag=True, help="Only index and parse the showcase dataset")
@click.option("--summarize", is_flag=True, help="Summarize the showcase thesaurus")
@click.option("--query-text", default=None, help="Question to ask over the indexed showcase")
def showcase_rag_cmd(
    input_location,
    output_location,
    parser,
    parse_method,
    query_mode,
    response_type,
    query_optimization_strategy,
    context_window,
    top_k,
    enable_vlm,
    index_only,
    summarize,
    query_text,
):
    """Index, summarize, visualize-ready summarize, and query the showcase dataset."""
    techniques = ShowcaseTechniques(
        parser=parser,
        parse_method=parse_method,
        query_mode=query_mode,
        response_type=response_type,
        query_optimization_strategy=query_optimization_strategy,
        context_window=context_window,
        enable_vlm=enable_vlm,
        top_k=top_k,
    )

    result = index_showcase_dataset_sync(
        input_location=input_location,
        output_location=output_location,
        techniques=techniques,
    )
    payload = {"indexing": result.to_dict()}

    if summarize:
        payload["summary"] = summarize_showcase_sync(
            input_location=input_location,
            techniques=techniques,
        )

    if query_text:
        payload["query"] = query_showcase_thesaurus_sync(
            query=query_text,
            input_location=input_location,
            techniques=techniques,
        )

    if index_only and not summarize and not query_text:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    cli()
