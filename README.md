# RAG Anything — Multimodal Document Processing & Retrieval Pipeline

A unified pipeline for multimodal document parsing, retrieval-augmented generation (RAG), and query optimization. Combines document parsing, multimodal content processing, hybrid retrieval, and evaluation in a single framework.

## What It Does

This project provides:

- **Multimodal document parsing** — Support for PDF, Office documents (via LibreOffice conversion), images, and text files using MinerU, Docling, and PaddleOCR parsers
- **Multimodal content processing** — Automatic detection and processing of images, tables, equations, and other content types with LLM-powered analysis and context-aware extraction
- **Production RAG pipeline** — Hybrid semantic + BM25 retrieval with cross-encoder reranking
- **Query optimization** — Query analysis, expansion, rewriting, and strategy suggestion
- **Evaluation framework** — Standard retrieval metrics (Precision@K, Recall@K, NDCG, MRR, MAP)
- **Processing callbacks** — Event system for observability and metrics collection
- **Batch processing** — Concurrent document processing with configurable workers

## Quick Start

### Installation

```bash
pip install -e .
pip install -e ".[dev,notebooks]"
```

### Jupyter Notebooks

```bash
jupyter lab notebooks/
```

## Pipeline Structure

The project follows a staged notebook structure:

| Stage | Notebook | Description |
|-------|----------|-------------|
| S00 | `s00_pipeline_setup.ipynb` | Pipeline initialization and configuration |
| S01 | `s01_document_ingestion.ipynb` | Document parsing and insertion |
| S02 | `s02_multimodal_processing.ipynb` | Multimodal content processing |
| S03 | `s03_query_optimization.ipynb` | Query analysis and optimization |
| S04 | `s04_evaluation.ipynb` | Retrieval evaluation |
| S05 | `s05_rag_demo.ipynb` | End-to-end demo |

## Usage

### Python API

```python
from aifluent import AIFluent, AIFluentConfig

# Initialize
config = AIFluentConfig(parser="mineru")
rag = AIFluent(config=config)

# Ingest documents
await rag.process_document_complete("path/to/document.pdf")

# Query
result = await rag.aquery("What is this document about?", mode="mix")
print(result)
```

### CLI

```bash
# Ingest documents
python -m src.cli ingest data/raw/sample.pdf --method auto

# Query
python -m src.cli query "What is the key finding?"

# Evaluate
python -m src.cli evaluate tests/test_queries.json

# Batch process
python -m src.cli batch data/raw/ --workers 4
```

## Project Structure

```
multimodal-analytics/
├── aifluent/
│   ├── __init__.py          # Main exports: AIFluent, AIFluentConfig
│   ├── core.py              # AIFluent main class
│   ├── config.py            # AIFluentConfig with env var support
│   ├── parser.py            # MinerU, Docling, PaddleOCR document parsers
│   ├── processor.py         # ProcessorMixin for document processing
│   ├── query.py             # QueryMixin for RAG querying
│   ├── batch.py             # BatchMixin for batch processing
│   ├── modalprocessors.py   # Image/Table/Equation/Generic processors
│   ├── prompt.py            # Prompt templates (PromptRegistry)
│   ├── callbacks.py         # ProcessingCallback event system
│   ├── utils.py             # Utility functions
│   ├── resilience.py        # Resilience utilities
│   ├── enhanced_markdown.py # Enhanced markdown processing
│   ├── prompts_zh.py        # Chinese prompt templates
│   ├── retrieval/           # Production RAG retrieval
│   │   ├── base.py          # BaseRetriever, RetrievalResult
│   │   ├── semantic.py      # SemanticRetriever
│   │   ├── bm25_retriever.py # BM25Retriever
│   │   ├── hybrid.py        # HybridRetriever (RRF fusion)
│   │   └── reranker.py      # CrossEncoderReranker
│   └── query/               # Query optimization
│       ├── analyzer.py      # QueryAnalyzer
│       └── optimizer.py     # QueryOptimizer
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # Precision@K, Recall@K, NDCG, MRR, MAP
│   └── evaluator.py         # Evaluator, EvaluationResult
├── src/
│   ├── __init__.py
│   ├── cli.py               # CLI entry points
│   ├── workflows.py         # Reusable workflow functions
│   ├── notebook_utils.py    # Shared notebook utilities
│   └── viz.py               # Visualization helpers
├── notebooks/               # Jupyter notebooks (s00–s05)
├── data/
│   ├── raw/
│   ├── processed/
│   └── checkpoints/
└── tests/
```

## Configuration

All configuration is done via `AIFluentConfig` or environment variables:

| Env Variable | Default | Description |
|---|---|---|
| `WORKING_DIR` | `./rag_storage` | RAG storage directory |
| `PARSER` | `mineru` | Parser (mineru, docling, paddleocr) |
| `PARSE_METHOD` | `auto` | Parse method (auto, ocr, txt) |
| `ENABLE_IMAGE_PROCESSING` | `True` | Enable image processing |
| `ENABLE_TABLE_PROCESSING` | `True` | Enable table processing |
| `ENABLE_EQUATION_PROCESSING` | `True` | Enable equation processing |
| `MAX_CONCURRENT_FILES` | `1` | Batch workers |
| `CONTEXT_WINDOW` | `1` | Context window for multimodal |

## Development

```bash
# Run tests
pytest

# Type checking
mypy aifluent/ evaluation/ src/

# Linting
ruff check .
```

## References

- **AIFluent**: Document parsing and multimodal processing from [AIFluent](https://github.com/kflow-dev/aifluent)
- **Production RAG**: Retrieval and evaluation from [production-rag](https://github.com/kflow-dev/production-rag)
- **Notebook structure**: Following the [artificial-data-generation](https://github.com/kflow-dev/artificial-data-generation) convention
