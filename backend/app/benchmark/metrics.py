"""
metrics.py — Benchmark metric calculations.

All metric computations are pure functions operating on lists of IDs.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Any


def recall_at_k(exact_ids: List[str], approx_ids: List[str], k: int = 10) -> float:
    """
    Recall@k: fraction of exact top-k results present in the approximate top-k.

    recall@k = |exact_top_k ∩ approx_top_k| / k

    Parameters
    ----------
    exact_ids : ordered list of exact nearest neighbor IDs (ground truth)
    approx_ids: ordered list of approximate neighbor IDs
    k         : cutoff

    Returns float in [0.0, 1.0].
    """
    if k == 0:
        return 0.0
    exact_set = set(exact_ids[:k])
    approx_set = set(approx_ids[:k])
    return len(exact_set & approx_set) / k


def aggregate_recalls(recalls: List[float]) -> Dict[str, float]:
    """Aggregate recall values into summary statistics."""
    if not recalls:
        return {
            "mean": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "std": 0.0
        }
    return {
        "mean": statistics.mean(recalls),
        "median": statistics.median(recalls),
        "min": min(recalls),
        "max": max(recalls),
        "std": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
    }


def aggregate_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    """Aggregate latency values (ms) into summary statistics."""
    if not latencies_ms:
        return {
            "mean_ms": 0.0, "median_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0
        }
    sorted_l = sorted(latencies_ms)
    p95_idx = int(0.95 * len(sorted_l))
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p95_ms": sorted_l[p95_idx],
    }


def compute_speedup(
    exact_latencies_ms: List[float], hnsw_latencies_ms: List[float]
) -> float:
    """Speedup = mean(exact_latency) / mean(hnsw_latency)."""
    if not exact_latencies_ms or not hnsw_latencies_ms:
        return 0.0
    mean_exact = statistics.mean(exact_latencies_ms)
    mean_hnsw = statistics.mean(hnsw_latencies_ms)
    if mean_hnsw < 1e-9:
        return float("inf")
    return mean_exact / mean_hnsw
