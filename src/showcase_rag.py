"""Parameterized showcase RAG workflow for notebooks and CLI."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.create_thesaurus import create_thesaurus


@dataclass
class ShowcaseTechniques:
    parser: str = "mineru"
    parse_method: str = "auto"
    query_mode: str = "mix"
    response_type: str = "Multiple Paragraphs"
    query_optimization_strategy: str = "all"
    context_window: int = 1
    enable_vlm: bool = False
    top_k: int = 10


@dataclass
class ShowcasePipelineResult:
    input_location: str
    techniques: dict
    thesaurus: dict
    ingestion: dict
    graph_summary: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_showcase_files(input_root: Path) -> list[str]:
    thesaurus = create_thesaurus(input_root)
    return [item["path"] for item in thesaurus["files"]]


async def index_showcase_dataset(
    input_location: str = "./data/raw/showcase",
    output_location: str | None = None,
    techniques: ShowcaseTechniques | None = None,
) -> ShowcasePipelineResult:
    techniques = techniques or ShowcaseTechniques()
    input_root = Path(input_location)
    thesaurus = create_thesaurus(input_root)
    file_paths = [item["path"] for item in thesaurus["files"]]

    from src.runtime import build_rag

    rag = build_rag()
    previous_parser = rag.config.parser
    previous_context_window = rag.config.context_window

    try:
        rag.config.parser = techniques.parser
        rag.config.context_window = techniques.context_window
        ingestion = await rag.process_documents_with_rag_batch(
            file_paths=file_paths,
            output_dir=output_location or rag.config.parser_output_dir,
            parse_method=techniques.parse_method,
            max_workers=1,
            recursive=True,
        )
        graph_summary = await extract_graph_summary(rag)
        return ShowcasePipelineResult(
            input_location=str(input_root),
            techniques=asdict(techniques),
            thesaurus=thesaurus,
            ingestion=ingestion,
            graph_summary=graph_summary,
        )
    finally:
        rag.config.parser = previous_parser
        rag.config.context_window = previous_context_window
        await rag._async_close()


async def extract_graph_summary(rag: Any, limit: int = 25) -> dict:
    graph = getattr(rag.lightrag, "chunk_entity_relation_graph", None) if rag.lightrag else None
    if graph is None:
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

    nodes: list[dict] = []
    edges: list[dict] = []

    try:
        if hasattr(graph, "get_all_labels"):
            labels = await graph.get_all_labels()
            for label in labels[:limit]:
                node_data = await graph.get_node(label)
                nodes.append({"id": label, **(node_data or {})})
        if hasattr(graph, "get_all_edges"):
            raw_edges = await graph.get_all_edges()
            for edge in raw_edges[:limit]:
                if isinstance(edge, dict):
                    edges.append(edge)
                elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                    edges.append({"source": edge[0], "target": edge[1]})
    except Exception as exc:
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "error": str(exc),
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


async def query_showcase_thesaurus(
    query: str,
    input_location: str = "./data/raw/showcase",
    techniques: ShowcaseTechniques | None = None,
    response_type: str | None = None,
    optimize: bool = True,
) -> dict:
    techniques = techniques or ShowcaseTechniques()
    from src.runtime import build_rag

    rag = build_rag()

    try:
        query_variants = [query]
        analysis = None
        if optimize:
            analysis = rag.query_analyzer.analyze(query)
            query_variants = rag.query_optimizer.optimize(
                query,
                strategy=techniques.query_optimization_strategy,
            )

        selected_query = query_variants[0]
        result = await rag.aquery(
            selected_query,
            mode=techniques.query_mode,
            response_type=response_type or techniques.response_type,
            top_k=techniques.top_k,
            vlm_enhanced=techniques.enable_vlm,
        )
        return {
            "query": query,
            "selected_query": selected_query,
            "query_variants": query_variants,
            "analysis": asdict(analysis) if analysis else None,
            "result": result,
        }
    finally:
        await rag._async_close()


async def summarize_showcase(
    input_location: str = "./data/raw/showcase",
    techniques: ShowcaseTechniques | None = None,
) -> dict:
    techniques = techniques or ShowcaseTechniques()
    thesaurus = create_thesaurus(Path(input_location))
    prompt = (
        "Summarize the document thesaurus by document types, sizes, likely entities, "
        "and likely cross-document relations. Use bullet points."
        f"\n\nThesaurus JSON:\n{json.dumps(thesaurus['thesaurus'])[:16000]}"
    )
    summary = await query_showcase_thesaurus(
        prompt,
        input_location=input_location,
        techniques=techniques,
        response_type="Bullet Points",
        optimize=False,
    )
    summary["thesaurus"] = thesaurus
    return summary


def index_showcase_dataset_sync(
    input_location: str = "./data/raw/showcase",
    output_location: str | None = None,
    techniques: ShowcaseTechniques | None = None,
) -> ShowcasePipelineResult:
    return asyncio.run(
        index_showcase_dataset(
            input_location=input_location,
            output_location=output_location,
            techniques=techniques,
        )
    )


def query_showcase_thesaurus_sync(
    query: str,
    input_location: str = "./data/raw/showcase",
    techniques: ShowcaseTechniques | None = None,
    response_type: str | None = None,
    optimize: bool = True,
) -> dict:
    return asyncio.run(
        query_showcase_thesaurus(
            query=query,
            input_location=input_location,
            techniques=techniques,
            response_type=response_type,
            optimize=optimize,
        )
    )


def summarize_showcase_sync(
    input_location: str = "./data/raw/showcase",
    techniques: ShowcaseTechniques | None = None,
) -> dict:
    return asyncio.run(
        summarize_showcase(
            input_location=input_location,
            techniques=techniques,
        )
    )
