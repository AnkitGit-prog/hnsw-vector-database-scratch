"""
build_index.py — Builds both ExactVectorIndex and HNSWIndex from the saved dataset.

Usage:
  python scripts/build_index.py [--M 16] [--ef_construction 200]
"""

from __future__ import annotations

import argparse
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


def main(M: int = 16, ef_construction: int = 200) -> None:
    from app.indexes.exact_index import ExactVectorIndex
    from app.indexes.hnsw_index import HNSWIndex
    from app.storage.persistence import load_vectors, save_hnsw_graph

    # Load dataset
    logger.info("Loading dataset from disk...")
    result = load_vectors("dataset")
    if result is None:
        logger.error("Dataset not found. Run generate_dataset.py first.")
        sys.exit(1)

    ids, vectors, metadata_list = result
    logger.info(f"Loaded {len(ids):,} vectors (dim={vectors.shape[1]})")

    # ---- Build Exact Index ----
    logger.info("\nBuilding ExactVectorIndex...")
    exact = ExactVectorIndex()
    t0 = time.perf_counter()
    for id_, vec, meta in zip(ids, vectors, metadata_list):
        exact.insert(id_, vec, meta)
    exact_time = time.perf_counter() - t0
    logger.info(f"Exact index built in {exact_time:.2f}s | {len(exact):,} vectors")

    # Quick sanity check
    test_vec = vectors[0]
    results = exact.search(test_vec, top_k=5)
    logger.info(f"Exact search sanity check: top result ID={results[0]['id']}, sim={results[0]['similarity']:.4f}")
    assert results[0]["id"] == ids[0], "Sanity check FAILED: top result should be itself!"
    logger.info("Exact index sanity check PASSED ✓")

    # ---- Build HNSW Index ----
    logger.info(f"\nBuilding HNSWIndex (M={M}, ef_construction={ef_construction})...")
    logger.info("This will take several minutes for 50,000 vectors...")
    hnsw = HNSWIndex(M=M, ef_construction=ef_construction, ef_search=50)

    t0 = time.perf_counter()
    for i, (id_, vec, meta) in enumerate(zip(ids, vectors, metadata_list)):
        hnsw.insert(id_, vec, meta)
        if (i + 1) % 5000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(ids) - i - 1) / rate
            logger.info(
                f"  Progress: {i+1:,}/{len(ids):,} "
                f"({100*(i+1)/len(ids):.1f}%) | "
                f"{elapsed:.0f}s elapsed | "
                f"~{remaining:.0f}s remaining"
            )

    hnsw_time = time.perf_counter() - t0
    stats = hnsw.stats()
    logger.info(
        f"HNSW index built in {hnsw_time:.2f}s | "
        f"{stats['active_vectors']:,} vectors | "
        f"max_layer={stats['max_layer']}"
    )

    # Quick HNSW sanity check
    hnsw_results = hnsw.search(test_vec, top_k=10)
    hnsw_ids = [r["id"] for r in hnsw_results]
    exact_results = exact.search(test_vec, top_k=10)
    exact_ids = [r["id"] for r in exact_results]
    from app.benchmark.metrics import recall_at_k
    recall = recall_at_k(exact_ids, hnsw_ids, k=10)
    logger.info(f"HNSW sanity check: recall@10 on 1 query = {recall:.2f}")

    # Save HNSW graph
    save_hnsw_graph(hnsw, name="hnsw")
    logger.info("HNSW graph saved.")

    logger.info("\n" + "=" * 60)
    logger.info("  Index Build Complete!")
    logger.info(f"  Exact build time : {exact_time:.2f}s")
    logger.info(f"  HNSW build time  : {hnsw_time:.2f}s")
    logger.info(f"  HNSW max layer   : {stats['max_layer']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    args = parser.parse_args()
    main(M=args.M, ef_construction=args.ef_construction)
