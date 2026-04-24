# AIFluent - Multimodal Document Processing & Retrieval Pipeline

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
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install -e ".[dev,notebooks]"
```

### Local End-to-End Run

The repository includes [`run_local.py`](./run_local.py) for a practical local flow using
LightRAG's OpenAI-compatible model adapters.

1. Create an environment file:

```bash
cp .env.example .env
```

2. Fill in at least `OPENAI_API_KEY` in `.env`

3. Load the environment and run:

```bash
set -a
source .env
set +a

python run_local.py ingest data/raw/sample.pdf
python run_local.py query "What is this document about?" --mode mix
```

You can also ingest and query in one step:

```bash
python run_local.py ingest-and-query data/raw/sample.pdf "Summarize the document"
```

Search + ingest from local folders, URL manifests, or a direct URL:

```bash
python run_local.py search-ingest \
  --keywords genai,rag \
  --document-types pdf,text \
  --input-location data/raw \
  --output-location ./output
```

Minimal web UI for the same workflow:

```bash
python run_local.py serve-ui --host 127.0.0.1 --port 8080
```

Index and query the local showcase dataset with parameterized techniques:

```bash
python run_local.py showcase-rag \
  --input-location ./data/raw/showcase \
  --output-location ./output/showcase \
  --parser mineru \
  --parse-method auto \
  --query-mode mix \
  --query-optimization-strategy all \
  --summarize \
  --query-text "What entities and relations are present in the showcase dataset?"
```

Notes:

- `run_local.py` expects `lightrag` and its OpenAI helper modules to be installed via project dependencies.
- If `lightrag` is not installed, `run_local.py` can also import it from `LIGHTRAG_SOURCE_DIR` and defaults to `./data/vendor/000_LightRAG`.
- `openai` is an explicit dependency here because the local runner uses `lightrag.llm.openai`.
- Parser-specific runtimes such as LibreOffice may still be required depending on file type and parser choice.
- `OPENAI_BASE_URL` can be used for OpenAI-compatible providers, not just OpenAI itself.

### Jupyter Notebooks

```bash
jupyter lab notebooks/
```

## Pipeline Structure

The project follows a staged notebook structure:

| Stage | Notebook | Description |
|-------|----------|-------------|
| S00 | `s00_pipeline_setup.ipynb` | Pipeline initialization and configuration |
| S00+ | `s00_create_sample_input_dataset.ipynb` | Uses OpenAI web search and/or Google Custom Search to discover and download sample assets into `./data/raw/showcase`, then builds a thesaurus summary |
| S01 | `s01_document_ingestion.ipynb` | Document parsing and insertion |
| S01+ | `s01_search_download_ingestion_showcase.ipynb` | Search, local download, ingestion, and thesaurus-style inventory showcase |
| S02+ | `s02_showcase_rag_pipeline.ipynb` | Indexes `./data/raw/showcase`, creates embeddings, extracts entities and relations, summarizes the thesaurus, visualizes graph/thesaurus snapshots, and runs Q&A |
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

# Query (works after LightRAG storage has been initialized)
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
