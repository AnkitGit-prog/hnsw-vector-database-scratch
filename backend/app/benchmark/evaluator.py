"""
evaluator.py — Benchmark evaluator.

Runs 500-query benchmark comparing exact vs HNSW search.
Computes recall@10, latency, speedup.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import (
    aggregate_latencies,
    aggregate_recalls,
    compute_speedup,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Runs the 500-query benchmark comparison between ExactVectorIndex and HNSWIndex.

    Ground truth is ALWAYS the ExactVectorIndex results.
    """

    def __init__(self, exact_index, hnsw_index, k: int = 10) -> None:
        self.exact_index = exact_index
        self.hnsw_index = hnsw_index
        self.k = k

    def run(
        self,
        query_vectors: np.ndarray,
        ef_search_override: Optional[int] = None,
        warmup_queries: int = 5,
    ) -> Dict[str, Any]:
        """
        Run benchmark over all query vectors.

        Parameters
        ----------
        query_vectors       : shape (Q, D) array of query embeddings
        ef_search_override  : override HNSW ef_search for this run
        warmup_queries      : number of warmup queries to run before timing

        Returns a comprehensive results dict.
        """
        n_queries = len(query_vectors)
        logger.info(
            f"Running benchmark: {n_queries} queries, k={self.k}, "
            f"ef_search={ef_search_override or self.hnsw_index.ef_search}"
        )

        # --- Warmup (exclude from timing) ---
        n_warmup = min(warmup_queries, n_queries)
        for i in range(n_warmup):
            self.exact_index.search(query_vectors[i], self.k)
            self.hnsw_index.search(query_vectors[i], self.k, ef_search_override)

        # --- Main benchmark loop ---
        per_query_results = []
        exact_latencies_ms = []
        hnsw_latencies_ms = []
        recalls = []

        for i, qvec in enumerate(query_vectors):
            # Exact search (ground truth)
            t0 = time.perf_counter()
            exact_results = self.exact_index.search(qvec, self.k)
            exact_ms = (time.perf_counter() - t0) * 1000

            # HNSW search
            t1 = time.perf_counter()
            hnsw_results = self.hnsw_index.search(qvec, self.k, ef_search_override)
            hnsw_ms = (time.perf_counter() - t1) * 1000

            # Compute recall@k
            exact_ids = [r["id"] for r in exact_results]
            hnsw_ids = [r["id"] for r in hnsw_results]
            r_at_k = recall_at_k(exact_ids, hnsw_ids, self.k)

            recalls.append(r_at_k)
            exact_latencies_ms.append(exact_ms)
            hnsw_latencies_ms.append(hnsw_ms)

            per_query_results.append(
                {
                    "query_index": i,
                    "exact_ids": exact_ids,
                    "hnsw_ids": hnsw_ids,
                    "recall_at_k": r_at_k,
                    "exact_latency_ms": exact_ms,
                    "hnsw_latency_ms": hnsw_ms,
                    "exact_top_similarities": [r["similarity"] for r in exact_results],
                    "hnsw_top_similarities": [r["similarity"] for r in hnsw_results],
                }
            )

        # --- Aggregate metrics ---
        recall_stats = aggregate_recalls(recalls)
        exact_lat_stats = aggregate_latencies(exact_latencies_ms)
        hnsw_lat_stats = aggregate_latencies(hnsw_latencies_ms)
        speedup = compute_speedup(exact_latencies_ms, hnsw_latencies_ms)

        exact_stats = self.exact_index.stats()
        hnsw_stats = self.hnsw_index.stats()

        results = {
            "summary": {
                "n_queries": n_queries,
                "k": self.k,
                "n_vectors": exact_stats["active_vectors"],
                "dimension": exact_stats["dimension"],
                "ef_search_used": ef_search_override or self.hnsw_index.ef_search,
                "hnsw_M": hnsw_stats["M"],
                "hnsw_ef_construction": hnsw_stats["ef_construction"],
                "recall": recall_stats,
                "exact_latency": exact_lat_stats,
                "hnsw_latency": hnsw_lat_stats,
                "speedup": speedup,
            },
            "per_query": per_query_results,
        }

        logger.info(
            f"Benchmark complete: recall@{self.k}={recall_stats['mean']:.3f}, "
            f"speedup={speedup:.2f}x, "
            f"exact={exact_lat_stats['mean_ms']:.2f}ms, "
            f"hnsw={hnsw_lat_stats['mean_ms']:.2f}ms"
        )
        return results

    def run_parameter_sweep(
        self,
        query_vectors: np.ndarray,
        ef_search_values: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the benchmark for each ef_search value (parameter experiment).

        Returns a list of summary dicts, one per ef_search value.
        This demonstrates the accuracy–latency tradeoff.
        """
        if ef_search_values is None:
            ef_search_values = [10, 25, 50, 100, 200]

        sweep_results = []
        for ef in ef_search_values:
            logger.info(f"Parameter sweep: ef_search={ef}")
            result = self.run(query_vectors, ef_search_override=ef)
            sweep_results.append(
                {
                    "ef_search": ef,
                    **result["summary"],
                }
            )

        return sweep_results
