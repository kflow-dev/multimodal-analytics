"""Evaluation metrics for retrieval systems"""

from typing import List, Set, Dict
from dataclasses import dataclass
import math


@dataclass
class MetricResult:
    """Retrieval metrics result"""
    precision_at_k: Dict[int, float]  # {1: 0.5, 5: 0.4, 10: 0.35}
    mrr: float                         # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]       # Normalized DCG
    map_score: float                   # Mean Average Precision
    recall_at_k: Dict[int, float]     # {1: 0.1, 5: 0.3, 10: 0.5}

    def __repr__(self):
        return (
            f"MetricResult(\n"
            f"  P@10={self.precision_at_k.get(10, 0):.3f}, "
            f"  MRR={self.mrr:.3f}, "
            f"  NDCG@10={self.ndcg_at_k.get(10, 0):.3f}, "
            f"  MAP={self.map_score:.3f}\n"
            f")"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "map": self.map_score,
            "recall_at_k": self.recall_at_k,
        }


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Precision@K: What % of top-k results are relevant?

    Precision@k = |relevant & retrieved_top_k| / k

    Args:
        retrieved_ids: List of retrieved document IDs (sorted by score)
        relevant_ids: Set of known relevant document IDs
        k: Consider top-k results

    Returns:
        Precision score 0.0-1.0
    """
    top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(top_k) & relevant_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K: What % of relevant items did we retrieve?

    Recall@k = |relevant & retrieved_top_k| / |relevant|

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of known relevant document IDs
        k: Consider top-k results

    Returns:
        Recall score 0.0-1.0
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k) & relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank: Where is the first relevant result?

    MRR = 1 / (position of first relevant document)

    Args:
        retrieved_ids: List of retrieved document IDs (sorted by score)
        relevant_ids: Set of known relevant document IDs

    Returns:
        MRR score 0.0-1.0
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0  # No relevant documents found


def compute_dcg(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
    """
    Discounted Cumulative Gain: Ranked retrieval with discount for position

    DCG = sum(relevance_i / log2(i + 1))

    Args:
        retrieved_ids: List of retrieved IDs (sorted by score)
        relevant_ids: Set of relevant IDs
        k: Consider top-k results

    Returns:
        DCG score
    """
    if not retrieved_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if k and rank > k:
            break

        relevance = 1.0 if doc_id in relevant_ids else 0.0
        discount = math.log2(rank + 1)
        dcg += relevance / discount

    return dcg


def compute_ideal_dcg(num_relevant: int, k: int = None) -> float:
    """
    Ideal DCG: Best possible DCG (all relevant docs at top)

    Args:
        num_relevant: Number of relevant documents
        k: Consider top-k

    Returns:
        Ideal DCG score
    """
    idcg = 0.0
    max_k = k if k else num_relevant

    for rank in range(1, max_k + 1):
        if rank <= num_relevant:
            discount = math.log2(rank + 1)
            idcg += 1.0 / discount
        else:
            break

    return idcg


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain: DCG normalized by ideal DCG

    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of known relevant document IDs
        k: Consider top-k results

    Returns:
        NDCG score 0.0-1.0
    """
    dcg = compute_dcg(retrieved_ids, relevant_ids, k)
    idcg = compute_ideal_dcg(len(relevant_ids), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_map(retrieved_ids_list: List[List[str]], relevant_ids_list: List[Set[str]]) -> float:
    """
    Mean Average Precision: Average precision across multiple queries

    MAP = (1/Q) * sum(AP_q for each query q)

    where AP_q = sum(P(k) * rel(k)) / num_relevant

    Args:
        retrieved_ids_list: List of retrieval results per query
        relevant_ids_list: List of relevant sets per query

    Returns:
        MAP score 0.0-1.0
    """
    aps = []

    for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
        if not relevant_ids:
            continue

        ap = 0.0
        num_relevant_found = 0

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                num_relevant_found += 1
                precision_at_rank = num_relevant_found / rank
                ap += precision_at_rank

        ap = ap / len(relevant_ids) if relevant_ids else 0.0
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def compute_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k_values: List[int] = None
) -> MetricResult:
    """
    Compute all retrieval metrics for single query.

    Args:
        retrieved_ids: List of retrieved document IDs (sorted by score)
        relevant_ids: Set of known relevant document IDs
        k_values: Evaluate at these k values

    Returns:
        MetricResult with all metrics
    """
    if k_values is None:
        k_values = [1, 5, 10]

    # Precision@K
    precision_at_k = {k: compute_precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values}

    # Recall@K
    recall_at_k = {k: compute_recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values}

    # MRR
    mrr = compute_mrr(retrieved_ids, relevant_ids)

    # NDCG@K
    ndcg_at_k = {k: compute_ndcg_at_k(retrieved_ids, relevant_ids, k) for k in k_values}

    # MAP (for single query, same as AP)
    map_score = ndcg_at_k.get(max(k_values)) if ndcg_at_k else 0.0

    return MetricResult(
        precision_at_k=precision_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        map_score=map_score,
        recall_at_k=recall_at_k
    )
