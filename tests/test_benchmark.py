"""
test_benchmark.py — Tests for benchmark metric calculations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from app.benchmark.metrics import (
    recall_at_k, aggregate_recalls, aggregate_latencies, compute_speedup
)


def test_recall_perfect():
    exact = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert recall_at_k(exact, exact, k=10) == 1.0


def test_recall_zero():
    exact = ["a", "b", "c"]
    approx = ["x", "y", "z"]
    assert recall_at_k(exact, approx, k=3) == 0.0


def test_recall_partial():
    exact = ["a", "b", "c", "d"]
    approx = ["a", "b", "x", "y"]
    assert abs(recall_at_k(exact, approx, k=4) - 0.5) < 1e-9


def test_recall_top_k_truncation():
    exact = ["a", "b", "c", "d", "e"]
    approx = ["a", "b", "c", "x", "y"]
    # Only top-3 are considered
    assert abs(recall_at_k(exact, approx, k=3) - 1.0) < 1e-9


def test_aggregate_recalls():
    recalls = [0.5, 0.6, 0.7, 0.8, 0.9]
    agg = aggregate_recalls(recalls)
    assert abs(agg["mean"] - 0.7) < 1e-9
    assert agg["min"] == 0.5
    assert agg["max"] == 0.9


def test_aggregate_recalls_empty():
    agg = aggregate_recalls([])
    assert agg["mean"] == 0.0


def test_speedup():
    speedup = compute_speedup([100.0, 200.0], [10.0, 20.0])
    assert abs(speedup - 10.0) < 1e-9


def test_speedup_zero_hnsw():
    speedup = compute_speedup([100.0], [0.0])
    assert speedup == float("inf")
