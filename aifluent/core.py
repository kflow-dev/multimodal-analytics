"""Core RAG Anything pipeline orchestrator - integrates parsing, retrieval, and querying."""

import os
import atexit
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from lightrag import LightRAG
from lightrag.utils import logger

from aifluent.config import AIFluentConfig
from aifluent.parser import get_parser, SUPPORTED_PARSERS
from aifluent.query import QueryMixin
from aifluent.processor import ProcessorMixin
from aifluent.batch import BatchMixin
from aifluent.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)
from aifluent.callbacks import CallbackManager

if TYPE_CHECKING:
    from aifluent.retrieval import HybridRetriever, SemanticRetriever, BM25Retriever
    from aifluent.query.optimizer import QueryOptimizer
    from aifluent.query.analyzer import QueryAnalyzer
    from aifluent.retrieval.reranker import CrossEncoderReranker


@dataclass
class AIFluent(QueryMixin, ProcessorMixin, BatchMixin):
    """Multimodal Document Processing Pipeline - Complete document parsing and RAG insertion."""

    # Core Components
    lightrag: Optional[LightRAG] = field(default=None)
    llm_model_func: Optional[Callable] = field(default=None)
    vision_model_func: Optional[Callable] = field(default=None)
    embedding_func: Optional[Callable] = field(default=None)
    config: Optional[AIFluentConfig] = field(default=None)

    # LightRAG Configuration
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Internal State
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    parse_cache: Optional[Any] = field(default=None, init=False)
    callback_manager: CallbackManager = field(
        default_factory=CallbackManager, init=False, repr=False
    )

    # Production RAG Pipeline Components
    retriever: Optional[Any] = field(default=None, init=False)
    reranker: Optional["CrossEncoderReranker"] = field(default=None, init=False)
    query_optimizer: Optional["QueryOptimizer"] = field(default=None, init=False)
    query_analyzer: Optional["QueryAnalyzer"] = field(default=None, init=False)

    # Evaluation
    _evaluation_results: Dict[str, Any] = field(default_factory=dict, init=False)
    _lightrag_storages_initialized: bool = field(
        default=False, init=False, repr=False
    )

    def __post_init__(self):
        if self.config is None:
            self.config = AIFluentConfig()

        self.working_dir = self.config.working_dir
        self.logger = logger
        self.doc_parser = get_parser(self.config.parser)

        atexit.register(self.close)

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        self.logger.info("AIFluent initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
                         f"Table: {self.config.enable_table_processing}, "
                         f"Equation: {self.config.enable_equation_processing}")
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

        # Initialize production RAG components
        self._init_production_components()

    def _init_production_components(self):
        """Initialize production RAG pipeline components."""
        from aifluent.retrieval.hybrid import HybridRetriever
        from aifluent.retrieval.reranker import CrossEncoderReranker
        from aifluent.query.optimizer import QueryOptimizer
        from aifluent.query.analyzer import QueryAnalyzer

        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.query_optimizer = QueryOptimizer()
        self.query_analyzer = QueryAnalyzer()

    def close(self):
        """Cleanup resources when object is destroyed."""
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    task = asyncio.create_task(self._async_close())
                    return
                else:
                    loop.run_until_complete(self._async_close())
                    loop.close()
            except RuntimeError:
                asyncio.run(self._async_close())
        except Exception:
            pass

    async def _async_close(self):
        """Async cleanup."""
        if self.lightrag and hasattr(self.lightrag, "close"):
            try:
                if hasattr(self.lightrag.close, "__await__"):
                    await self.lightrag.close()
                else:
                    self.lightrag.close()
            except Exception:
                pass

    async def _ensure_lightrag_initialized(self) -> None:
        """Ensure LightRAG is initialized."""
        if self.lightrag is None:
            self.lightrag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=self.embedding_func,
                **self.lightrag_kwargs,
            )

        if not self._lightrag_storages_initialized:
            if hasattr(self.lightrag, "initialize_storages"):
                await self.lightrag.initialize_storages()
            if hasattr(self.lightrag, "initialize_pipeline_status"):
                await self.lightrag.initialize_pipeline_status()
            self._lightrag_storages_initialized = True

        self.logger.info("LightRAG instance initialized")

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information as a dictionary."""
        config_info = {
            "parser": self.config.parser,
            "parse_method": self.config.parse_method,
            "working_dir": self.config.working_dir,
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
        }

        if self.lightrag_kwargs:
            safe_kwargs = {
                k: v for k, v in self.lightrag_kwargs.items()
                if not callable(v) and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {"custom_parameters": safe_kwargs}

        return config_info

    def set_content_source_for_context(self, content_source, content_format: str = "auto"):
        """Set content source for context extraction in all modal processors."""
        if not self.modal_processors:
            return
        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
            except Exception as e:
                self.logger.error(f"Failed to set content source for {processor_name}: {e}")

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration."""
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information."""
        from aifluent.utils import get_processor_supports
        from aifluent.parser import MineruParser

        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "parser_installation": {
                parser_name: get_parser(parser_name).check_installation()
                for parser_name in SUPPORTED_PARSERS
            },
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function" if self.llm_model_func else "Not provided",
                "vision_model": "External function" if self.vision_model_func else "Not provided",
                "embedding_model": "External function" if self.embedding_func else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
        else:
            base_info["status"] = "Initialized"

        return base_info

    async def evaluate(self, test_queries: List[tuple], k_values: List[int] = None) -> Dict[str, Any]:
        """Evaluate RAG pipeline on test queries.

        Args:
            test_queries: List of (query, relevant_doc_ids) tuples
            k_values: k values for metrics

        Returns:
            Evaluation results dictionary
        """
        from aifluent.retrieval.base import RetrievalResult

        if k_values is None:
            k_values = [1, 5, 10]

        all_query_metrics = []

        for query, relevant_ids_set in test_queries:
            relevant_ids = set(relevant_ids_set) if not isinstance(relevant_ids_set, set) else relevant_ids_set

            # Retrieve
            if self.retriever and hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query, k=max(k_values))
                retrieved_ids = [r.document_id for r in results]
            else:
                # Fallback: use LightRAG for retrieval
                await self._ensure_lightrag_initialized()
                query_param = type('QueryParam', (), {'mode': 'hybrid'})()
                raw_results = await self.lightrag.aquery(query, param=query_param)
                retrieved_ids = []

            # Compute metrics
            from evaluation.metrics import compute_metrics
            metrics = compute_metrics(retrieved_ids, relevant_ids, k_values)
            all_query_metrics.append({
                "query": query,
                "retrieved": retrieved_ids[:10],
                "relevant": list(relevant_ids)[:10],
                "metrics": metrics.to_dict()
            })

        # Average metrics
        avg_metrics = {}
        for k in k_values:
            precisions = [m["metrics"]["precision_at_k"].get(k, 0) for m in all_query_metrics]
            recalls = [m["metrics"]["recall_at_k"].get(k, 0) for m in all_query_metrics]
            ndcgs = [m["metrics"]["ndcg_at_k"].get(k, 0) for m in all_query_metrics]
            avg_metrics[f"precision_at_{k}"] = sum(precisions) / len(precisions) if precisions else 0
            avg_metrics[f"recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0
            avg_metrics[f"ndcg_at_{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0

        mrrs = [m["metrics"]["mrr"] for m in all_query_metrics]
        avg_metrics["mrr"] = sum(mrrs) / len(mrrs) if mrrs else 0
        avg_metrics["map"] = sum(m["metrics"]["map_score"] for m in all_query_metrics) / len(all_query_metrics) if all_query_metrics else 0

        self._evaluation_results = {
            "num_queries": len(test_queries),
            "per_query_metrics": all_query_metrics,
            "avg_metrics": avg_metrics,
            "k_values": k_values,
        }
        return self._evaluation_results

    def print_evaluation_comparison(self, results: Dict[str, Any]):
        """Print formatted evaluation comparison."""
        print(f"\n{'='*70}")
        print("RAG EVALUATION RESULTS")
        print(f"{'='*70}\n")
        print(f"Queries evaluated: {results['num_queries']}")
        print(f"\n{'Retriever':<25} {'P@10':>8} {'R@10':>8} {'NDCG@10':>10} {'MRR':>8} {'MAP':>8}")
        print("-" * 70)

        avg = results['avg_metrics']
        p10 = avg.get('precision_at_10', 0)
        r10 = avg.get('recall_at_10', 0)
        ndcg10 = avg.get('ndcg_at_10', 0)
        mrr = avg.get('mrr', 0)
        map_score = avg.get('map', 0)

        print(f"{'AIFluent':<25} {p10:>8.3f} {r10:>8.3f} {ndcg10:>10.3f} {mrr:>8.3f} {map_score:>8.3f}")
        print()

    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self._evaluation_results, f, indent=2, default=str)
        print(f"Evaluation results saved to {filepath}")
