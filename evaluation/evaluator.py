"""Evaluation runner for comparing retrievers"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import json
from .metrics import compute_metrics, MetricResult


@dataclass
class EvaluationResult:
    """Results for evaluating a single retriever"""
    retriever_name: str
    num_queries: int
    avg_metrics: Dict[str, float]
    per_query_metrics: List[Dict] = field(default_factory=list)

    def __repr__(self):
        return (
            f"{self.retriever_name}:\n"
            f"  Precision@10: {self.avg_metrics.get('precision_at_10', 0):.3f}\n"
            f"  Recall@10: {self.avg_metrics.get('recall_at_10', 0):.3f}\n"
            f"  MRR: {self.avg_metrics.get('mrr', 0):.3f}\n"
            f"  NDCG@10: {self.avg_metrics.get('ndcg_at_10', 0):.3f}\n"
        )


class Evaluator:
    """
    Evaluation framework for RAG systems.

    Runs benchmarks and compares retriever quality.
    """

    def __init__(self):
        self.results_cache = {}

    def evaluate(
        self,
        retriever,
        test_queries: List[Tuple[str, Set[str]]],
        k_values: List[int] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Evaluate retriever on test queries.

        Args:
            retriever: Retriever object with retrieve() method
            test_queries: List of (query, relevant_ids_set) tuples
            k_values: Evaluate at these k values (default: [1, 5, 10])
            verbose: Print progress

        Returns:
            EvaluationResult with metrics
        """
        if k_values is None:
            k_values = [1, 5, 10]

        metrics_list = []
        per_query_results = []

        for query_idx, (query, relevant_ids) in enumerate(test_queries):
            if verbose and (query_idx + 1) % 10 == 0:
                print(f"  Evaluated {query_idx + 1}/{len(test_queries)} queries...")

            # Retrieve documents
            results = retriever.retrieve(query, k=max(k_values))
            retrieved_ids = [r.document_id for r in results]

            # Compute metrics
            metrics = compute_metrics(retrieved_ids, relevant_ids, k_values)
            metrics_list.append(metrics)

            # Store per-query results
            per_query_results.append({
                "query": query,
                "retrieved": retrieved_ids[:10],
                "relevant": list(relevant_ids),
                "metrics": metrics.to_dict()
            })

        # Average metrics across queries
        avg_metrics = self._average_metrics(metrics_list, k_values)

        result = EvaluationResult(
            retriever_name=retriever.name,
            num_queries=len(test_queries),
            avg_metrics=avg_metrics,
            per_query_metrics=per_query_results
        )

        if verbose:
            print(f"\n{result}")

        return result

    def compare(
        self,
        retrievers: List,
        test_queries: List[Tuple[str, Set[str]]],
        k_values: List[int] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple retrievers.

        Args:
            retrievers: List of retriever objects
            test_queries: Test set
            k_values: Evaluation k values

        Returns:
            Dictionary mapping retriever name to EvaluationResult
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {len(retrievers)} retrievers on {len(test_queries)} queries")
        print(f"{'='*60}\n")

        results = {}

        for retriever in retrievers:
            print(f"Evaluating: {retriever.name}")
            eval_result = self.evaluate(retriever, test_queries, k_values, verbose=True)
            results[retriever.name] = eval_result

        return results

    def _average_metrics(self, metrics_list: List[MetricResult], k_values: List[int]) -> Dict:
        """Average metrics across multiple queries"""
        avg = {}

        # Average precision@k
        for k in k_values:
            precisions = [m.precision_at_k.get(k, 0) for m in metrics_list]
            avg[f"precision_at_{k}"] = sum(precisions) / len(precisions) if precisions else 0

            recalls = [m.recall_at_k.get(k, 0) for m in metrics_list]
            avg[f"recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0

            ndcgs = [m.ndcg_at_k.get(k, 0) for m in metrics_list]
            avg[f"ndcg_at_{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0

        # Average MRR and MAP
        mrrs = [m.mrr for m in metrics_list]
        avg["mrr"] = sum(mrrs) / len(mrrs) if mrrs else 0

        maps = [m.map_score for m in metrics_list]
        avg["map"] = sum(maps) / len(maps) if maps else 0

        return avg

    def print_comparison(self, results: Dict[str, EvaluationResult]):
        """Print formatted comparison"""
        print(f"\n{'='*70}")
        print("RETRIEVER COMPARISON")
        print(f"{'='*70}\n")

        # Header
        print(f"{'Retriever':<25} {'P@10':>8} {'R@10':>8} {'NDCG@10':>10} {'MRR':>8}")
        print("-" * 70)

        # Results
        for name, result in results.items():
            p10 = result.avg_metrics.get("precision_at_10", 0)
            r10 = result.avg_metrics.get("recall_at_10", 0)
            ndcg10 = result.avg_metrics.get("ndcg_at_10", 0)
            mrr = result.avg_metrics.get("mrr", 0)

            print(f"{name:<25} {p10:>8.3f} {r10:>8.3f} {ndcg10:>10.3f} {mrr:>8.3f}")

        print()

    def save_results(self, results: Dict[str, EvaluationResult], filepath: str):
        """Save evaluation results to JSON"""
        data = {}
        for name, result in results.items():
            data[name] = {
                "num_queries": result.num_queries,
                "avg_metrics": result.avg_metrics,
                "per_query_metrics": result.per_query_metrics[:3],  # Store first 3 for brevity
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")
