"""Visualization helpers for RAG Anything pipelines."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import numpy as np


def plot_retrieval_comparison(
    results: Dict[str, Dict[str, float]],
    metric_keys: List[str] = None,
    title: str = "Retriever Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison of retrieval strategies across metrics.

    Args:
        results: {retriever_name: {metric_name: value}}
        metric_keys: Metrics to plot
        title: Plot title
        save_path: Optional file to save the plot

    Returns:
        matplotlib Figure
    """
    if metric_keys is None:
        metric_keys = ["precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"]

    retrievers = list(results.keys())
    metrics = [k.replace("precision_at_", "P@").replace("recall_at_", "R@").replace("ndcg_at_", "NDCG@") for k in metric_keys]

    x = np.arange(len(retrievers))
    width = 0.8 / len(metric_keys)

    fig, ax = plt.subplots(figsize=(10 * len(retrievers) / 4, 6))

    for i, key in enumerate(metric_keys):
        values = [results[r].get(key, 0) for r in retrievers]
        ax.bar(x + i * width - (len(metric_keys) - 1) * width / 2, values, width, label=metrics[i])

    ax.set_xticks(x)
    ax.set_xticklabels(retrievers, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_evaluation_bar(
    avg_metrics: Dict[str, float],
    title: str = "Evaluation Results",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot average evaluation metrics as a bar chart.

    Args:
        avg_metrics: {metric_name: value}
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    labels = [k.replace("precision_at_", "P@").replace("recall_at_", "R@")
              .replace("ndcg_at_", "NDCG@") for k in avg_metrics.keys()]
    values = list(avg_metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(values))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Score")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_processing_timeline(
    timeline_events: List[Dict[str, str | float]],
    title: str = "Processing Timeline",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot document processing timeline.

    Args:
        timeline_events: List of {name, duration} dicts
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    names = [e.get("name", "unknown") for e in timeline_events]
    durations = [e.get("duration", 0) for e in timeline_events]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(names, durations, color=sns.color_palette("viridis", len(names)))

    for bar, dur in zip(bars, durations):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{dur:.2f}s", ha="left", va="center", fontsize=10)

    ax.set_xlabel("Duration (seconds)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
