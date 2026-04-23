"""Placeholder tests — extend with your test cases."""

import pytest


def test_placeholder():
    """Verify tests run."""
    assert True


def test_aifluent_config():
    """Verify AIFluentConfig loads with defaults."""
    from aifluent import AIFluentConfig
    config = AIFluentConfig()
    assert config.parser in ("mineru", "docling", "paddleocr")
    assert config.working_dir is not None


def test_evaluation_metrics():
    """Verify metrics computation."""
    from evaluation.metrics import compute_metrics
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}
    result = compute_metrics(retrieved, relevant, k_values=[1, 5])
    assert result.precision_at_k[1] == 1.0
    assert result.precision_at_k[5] == 3 / 5
    assert result.recall_at_k[5] == 1.0
