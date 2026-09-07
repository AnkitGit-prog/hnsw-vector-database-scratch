"""
index_service.py — Wires together both indexes, embedder, and persistence.

This is the single source of truth for the application state.
It is loaded once at startup and reused across all API requests.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..embeddings.embedder import Embedder
from ..indexes.exact_index import ExactVectorIndex
from ..indexes.hnsw_index import HNSWIndex
from ..benchmark.evaluator import BenchmarkEvaluator
from ..storage.persistence import (
    load_vectors,
    save_vectors,
    load_benchmark_results,
    save_benchmark_results,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"


class IndexService:
    """
    Application-level service: manages both indexes and the embedder.

    Lifecycle:
      1. Created at startup.
      2. Loads dataset from disk if available.
      3. Builds indexes from the dataset.
      4. Exposes search/insert/delete through a clean interface.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.embedder = Embedder()
        self.exact_index = ExactVectorIndex()
        self.hnsw_index = HNSWIndex(
            M=M, ef_construction=ef_construction, ef_search=ef_search
        )
        self._initialized = False
        self._build_time_exact = 0.0
        self._build_time_hnsw = 0.0
        self._cached_benchmark: Optional[Dict[str, Any]] = None
        self._cached_sweep: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, prefix: str = "dataset") -> bool:
        """
        Load dataset from disk and build both indexes.

        Returns True if data was found and loaded, False otherwise.
        """
        result = load_vectors(prefix)
        if result is None:
            logger.info("No saved dataset on disk. Initializing sample 1,000 vector index in memory...")
            rng = np.random.default_rng(42)
            dim = self.embedder.dimension
            topics = [
                "Artificial intelligence and machine learning algorithms",
                "Database indexing, graph search and vector search techniques",
                "Computer networking, TCP/IP protocol and web development",
                "Space exploration, rockets, astronomy and planetary science",
                "Quantum physics, relativity and theoretical mechanics",
            ]
            ids = []
            vectors = []
            metadata_list = []
            for i in range(1000):
                topic = topics[i % len(topics)]
                text = f"{topic} sample document entry #{i+1}"
                vec = rng.standard_normal(dim).astype(np.float32)
                norm = np.linalg.norm(vec)
                vec = vec / (norm if norm > 1e-10 else 1.0)
                ids.append(f"sample_{i:04d}")
                vectors.append(vec)
                metadata_list.append({"text": text, "source": "sample_seed", "topic": topic})
        else:
            ids, vectors, metadata_list = result
        logger.info(f"Building indexes from {len(ids)} vectors...")

        # Build exact index
        t0 = time.perf_counter()
        for id_, vec, meta in zip(ids, vectors, metadata_list):
            try:
                self.exact_index.insert(id_, vec, meta)
            except ValueError:
                pass  # skip duplicates gracefully
        self._build_time_exact = time.perf_counter() - t0
        logger.info(f"Exact index built in {self._build_time_exact:.2f}s")

        # Build HNSW index
        t0 = time.perf_counter()
        for id_, vec, meta in zip(ids, vectors, metadata_list):
            try:
                self.hnsw_index.insert(id_, vec, meta)
            except ValueError:
                pass
        self._build_time_hnsw = time.perf_counter() - t0
        logger.info(f"HNSW index built in {self._build_time_hnsw:.2f}s")

        # Load cached benchmark if available
        cached = load_benchmark_results("benchmark")
        if cached:
            self._cached_benchmark = cached
            logger.info("Loaded cached benchmark results.")

        cached_sweep = load_benchmark_results("sweep")
        if cached_sweep:
            self._cached_sweep = cached_sweep
            logger.info("Loaded cached sweep results.")

        self._initialized = True
        return True

    # ------------------------------------------------------------------
    # Text → embedding → search pipeline
    # ------------------------------------------------------------------

    def text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        return self.embedder.embed(text)

    def search_exact(
        self, query: Any, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Exact search. Query can be a text string or raw vector list.
        """
        vec = self._resolve_query(query)
        t0 = time.perf_counter()
        results = self.exact_index.search(vec, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"results": results, "latency_ms": latency_ms}

    def search_hnsw(
        self, query: Any, top_k: int = 10, ef_search: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        HNSW approximate search. Query can be text or raw vector.
        """
        vec = self._resolve_query(query)
        t0 = time.perf_counter()
        results = self.hnsw_index.search(vec, top_k, ef_search)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"results": results, "latency_ms": latency_ms}

    def insert(self, id_: str, text: str, extra_meta: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Embed text and insert into both indexes.
        """
        vec = self.embedder.embed(text)
        metadata = {
            "text": text,
            "source": "user_insert",
            **(extra_meta or {}),
        }
        self.exact_index.insert(id_, vec, metadata)
        self.hnsw_index.insert(id_, vec, metadata)
        return {"id": id_, "dimension": len(vec), "status": "inserted"}

    def insert_vector(
        self, id_: str, vector: List[float], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Insert a pre-computed vector into both indexes."""
        vec = np.array(vector, dtype=np.float32)
        self.exact_index.insert(id_, vec, metadata)
        self.hnsw_index.insert(id_, vec, metadata)
        return {"id": id_, "dimension": len(vec), "status": "inserted"}

    def delete(self, id_: str) -> Dict[str, Any]:
        """Tombstone-delete from both indexes."""
        exact_ok = self.exact_index.delete(id_)
        hnsw_ok = self.hnsw_index.delete(id_)
        if not exact_ok and not hnsw_ok:
            return {"status": "not_found", "id": id_}
        return {"status": "deleted", "id": id_}

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector by ID."""
        return self.exact_index.get(id_)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        n_queries: int = 500,
        k: int = 10,
        seed: int = 42,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the 500-query benchmark. Uses cached results unless force=True.
        """
        if self._cached_benchmark and not force:
            return self._cached_benchmark

        # Sample benchmark query vectors from the exact index
        rng = np.random.default_rng(seed)
        all_ids = self.exact_index.get_all_ids()
        if len(all_ids) < n_queries:
            query_ids = all_ids
        else:
            query_ids = rng.choice(all_ids, size=n_queries, replace=False).tolist()

        # Retrieve stored normalized vectors for benchmark queries
        query_vectors = np.array(
            [self.exact_index.get_normalized_vector(id_) for id_ in query_ids],
            dtype=np.float32,
        )

        evaluator = BenchmarkEvaluator(self.exact_index, self.hnsw_index, k=k)
        results = evaluator.run(query_vectors)
        results["summary"]["build_time_exact_s"] = self._build_time_exact
        results["summary"]["build_time_hnsw_s"] = self._build_time_hnsw

        save_benchmark_results(results, "benchmark")
        self._cached_benchmark = results
        return results

    def run_parameter_sweep(
        self,
        ef_search_values: Optional[List[int]] = None,
        n_queries: int = 200,
        seed: int = 42,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run ef_search parameter sweep benchmark."""
        if self._cached_sweep and not force:
            return self._cached_sweep

        if ef_search_values is None:
            ef_search_values = [10, 25, 50, 100, 200]

        rng = np.random.default_rng(seed)
        all_ids = self.exact_index.get_all_ids()
        query_ids = rng.choice(all_ids, size=min(n_queries, len(all_ids)), replace=False).tolist()
        query_vectors = np.array(
            [self.exact_index.get_normalized_vector(id_) for id_ in query_ids],
            dtype=np.float32,
        )

        evaluator = BenchmarkEvaluator(self.exact_index, self.hnsw_index, k=10)
        sweep = evaluator.run_parameter_sweep(query_vectors, ef_search_values)

        save_benchmark_results(sweep, "sweep")
        self._cached_sweep = sweep
        return sweep

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        exact_s = self.exact_index.stats()
        hnsw_s = self.hnsw_index.stats()
        exact_s["build_time_seconds"] = self._build_time_exact
        hnsw_s["build_time_seconds"] = self._build_time_hnsw
        return {
            "initialized": self._initialized,
            "exact": exact_s,
            "hnsw": hnsw_s,
            "embedder": self.embedder.stats(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_query(self, query: Any) -> np.ndarray:
        """Convert text or vector list to numpy array."""
        if isinstance(query, str):
            return self.embedder.embed(query)
        return np.array(query, dtype=np.float32)
