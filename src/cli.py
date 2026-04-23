"""CLI entry points for RAG Anything."""

import asyncio
import json
import click
from pathlib import Path


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
    from aifluent import AIFluent, AIFluentConfig

    config = AIFluentConfig(parser=parser)
    rag = AIFluent(config=config)

    async def _ingest():
        results = {"files": [], "errors": [], "duration": 0}
        import time
        start = time.time()

        for fp in file_paths:
            try:
                if Path(fp).is_dir():
                    await rag.process_folder_complete(
                        fp, output_dir=output, parse_method=method,
                        max_workers=workers, display_stats=stats,
                        recursive=recursive,
                    )
                else:
                    await rag.process_document_complete(
                        fp, output_dir=output, parse_method=method,
                        display_stats=stats,
                    )
                results["files"].append(fp)
            except Exception as e:
                results["errors"].append({"file": fp, "error": str(e)})

        results["duration"] = time.time() - start
        return results

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
    from aifluent import AIFluent, AIFluentConfig

    config = AIFluentConfig()
    rag = AIFluent(config=config)

    async def _query():
        return await rag.aquery(query, mode=mode)

    result = asyncio.run(_query())
    click.echo(result)


@cli.command(name="evaluate")
@click.argument("test_file", help="JSON file with test queries: [{\"query\": \"...\", \"relevant_ids\": [...]}]")
@click.option("--k-values", default="1,5,10", help="Comma-separated k values")
@click.option("--output", "-o", default=None, help="Output JSON file for results")
def evaluate_cmd(test_file, k_values, output):
    """Evaluate RAG pipeline on test queries."""
    with open(test_file) as f:
        test_cases = json.load(f)

    test_queries = [(tc["query"], set(tc.get("relevant_ids", []))) for tc in test_cases]
    k_values_list = [int(k) for k in k_values.split(",")]

    from aifluent import AIFluent, AIFluentConfig
    config = AIFluentConfig()
    rag = AIFluent(config=config)

    results = asyncio.run(rag.evaluate(test_queries, k_values=k_values_list))
    rag.print_evaluation_comparison(results)

    if output:
        rag.save_evaluation_results(output)


@cli.command()
@click.argument("file_paths", nargs=-1)
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--workers", "-w", default=1, help="Max concurrent workers")
@click.option("--recursive", "-r", is_flag=True, help="Process recursively")
def batch(file_paths, output, workers, recursive):
    """Process multiple documents in batch mode."""
    from aifluent import AIFluent, AIFluentConfig

    config = AIFluentConfig()
    rag = AIFluent(config=config)

    async def _batch():
        return await rag.process_documents_with_rag_batch(
            file_paths=file_paths,
            output_dir=output,
            max_workers=workers,
            recursive=recursive,
        )

    result = asyncio.run(_batch())
    click.echo(f"Batch complete: {result}")


@cli.command()
@click.option("--check", is_flag=True, help="Check parser installation")
def status(check):
    """Show RAG Anything status and configuration."""
    from aifluent import AIFluent, AIFluentConfig
    config = AIFluentConfig()
    rag = AIFluent(config=config)

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


if __name__ == "__main__":
    cli()
