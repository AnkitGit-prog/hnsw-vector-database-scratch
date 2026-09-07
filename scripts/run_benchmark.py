"""
run_benchmark.py — Runs the 500-query benchmark and prints results.

Usage:
  python scripts/run_benchmark.py [--n_queries 500] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main(n_queries: int = 500, k: int = 10, seed: int = 42) -> None:
    from app.indexes.exact_index import ExactVectorIndex
    from app.indexes.hnsw_index import HNSWIndex
    from app.benchmark.evaluator import BenchmarkEvaluator
    from app.storage.persistence import load_vectors, save_benchmark_results

    # Load dataset
    logger.info("Loading dataset...")
    result = load_vectors("dataset")
    if result is None:
        logger.error("Dataset not found. Run generate_dataset.py first.")
        sys.exit(1)

    ids, vectors, metadata_list = result
    logger.info(f"Loaded {len(ids):,} vectors (dim={vectors.shape[1]})")

    # Build indexes
    logger.info("Building ExactVectorIndex...")
    exact = ExactVectorIndex()
    for id_, vec, meta in zip(ids, vectors, metadata_list):
        exact.insert(id_, vec, meta)

    logger.info("Building HNSWIndex (M=16, ef_construction=200)...")
    hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=50)
    t0 = time.perf_counter()
    for i, (id_, vec, meta) in enumerate(zip(ids, vectors, metadata_list)):
        hnsw.insert(id_, vec, meta)
        if (i + 1) % 10000 == 0:
            logger.info(f"  {i+1:,}/{len(ids):,} inserted...")
    hnsw_build_time = time.perf_counter() - t0
    logger.info(f"HNSW built in {hnsw_build_time:.2f}s")

    # Sample benchmark queries
    rng = np.random.default_rng(seed)
    n_q = min(n_queries, len(ids))
    query_indices = rng.choice(len(ids), size=n_q, replace=False)
    query_vectors = vectors[query_indices]

    # ---- Main benchmark ----
    logger.info(f"\nRunning {n_q}-query benchmark (k={k})...")
    evaluator = BenchmarkEvaluator(exact, hnsw, k=k)
    results = evaluator.run(query_vectors)

    s = results["summary"]
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Vectors       : {s['n_vectors']:,}")
    print(f"  Dimension     : {s['dimension']}")
    print(f"  Queries       : {s['n_queries']}")
    print(f"  k             : {s['k']}")
    print(f"  HNSW M        : {s['hnsw_M']}")
    print(f"  ef_construction: {s['hnsw_ef_construction']}")
    print(f"  ef_search     : {s['ef_search_used']}")
    print()
    print(f"  Recall@{k}:")
    r = s["recall"]
    print(f"    Mean    : {r['mean']:.4f}")
    print(f"    Median  : {r['median']:.4f}")
    print(f"    Min     : {r['min']:.4f}")
    print(f"    Max     : {r['max']:.4f}")
    print()
    print(f"  Exact search latency:")
    e = s["exact_latency"]
    print(f"    Mean    : {e['mean_ms']:.2f} ms")
    print(f"    Median  : {e['median_ms']:.2f} ms")
    print(f"    P95     : {e['p95_ms']:.2f} ms")
    print()
    print(f"  HNSW search latency:")
    h = s["hnsw_latency"]
    print(f"    Mean    : {h['mean_ms']:.2f} ms")
    print(f"    Median  : {h['median_ms']:.2f} ms")
    print(f"    P95     : {h['p95_ms']:.2f} ms")
    print()
    print(f"  Speedup (exact/HNSW): {s['speedup']:.2f}x")
    print("=" * 60)

    # ---- Parameter sweep ----
    logger.info("\nRunning ef_search parameter sweep...")
    sweep_results = evaluator.run_parameter_sweep(
        query_vectors[:200],
        ef_search_values=[10, 25, 50, 100, 200],
    )

    print("\n  ef_search Parameter Sweep:")
    print(f"  {'ef_search':>10} | {'Recall@10':>10} | {'Avg Latency':>12} | {'Speedup':>8}")
    print("  " + "-" * 50)
    for row in sweep_results:
        ef = row["ef_search"]
        recall_mean = row["recall"]["mean"]
        hnsw_ms = row["hnsw_latency"]["mean_ms"]
        speedup = row["speedup"]
        print(f"  {ef:>10} | {recall_mean:>10.4f} | {hnsw_ms:>10.2f}ms | {speedup:>7.2f}x")

    # Save results
    save_benchmark_results(results, "benchmark")
    save_benchmark_results(sweep_results, "sweep")
    logger.info("\nBenchmark results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_queries=args.n_queries, k=args.k, seed=args.seed)
