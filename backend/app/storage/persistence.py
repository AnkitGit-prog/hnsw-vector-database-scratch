"""
persistence.py — File-based persistence for vector indexes.

Saves/loads:
  - Vectors as .npz (compressed NumPy)
  - Metadata as JSON
  - HNSW graph structure as JSON (adjacency lists + node info)
  - Benchmark results as JSON
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
BENCHMARK_DIR = DATA_DIR / "benchmarks"


def ensure_dirs() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def save_vectors(
    ids: List[str],
    vectors: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    prefix: str = "dataset",
) -> None:
    """Save vectors to .npz and metadata to JSON."""
    ensure_dirs()
    np_path = GENERATED_DIR / f"{prefix}_vectors.npz"
    meta_path = GENERATED_DIR / f"{prefix}_metadata.json"

    np.savez_compressed(str(np_path), vectors=vectors, ids=np.array(ids))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False)

    logger.info(f"Saved {len(ids)} vectors to {np_path}")


def load_vectors(
    prefix: str = "dataset",
) -> Optional[Tuple[List[str], np.ndarray, List[Dict[str, Any]]]]:
    """Load vectors from .npz and metadata from JSON. Returns None if not found."""
    np_path = GENERATED_DIR / f"{prefix}_vectors.npz"
    meta_path = GENERATED_DIR / f"{prefix}_metadata.json"

    if not np_path.exists() or not meta_path.exists():
        return None

    data = np.load(str(np_path), allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    ids = data["ids"].tolist()

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    logger.info(f"Loaded {len(ids)} vectors from {np_path}")
    return ids, vectors, metadata_list


def save_benchmark_results(results: Dict[str, Any], name: str = "benchmark") -> None:
    """Save benchmark results to JSON."""
    ensure_dirs()
    path = BENCHMARK_DIR / f"{name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved benchmark results to {path}")


def load_benchmark_results(name: str = "benchmark") -> Optional[Dict[str, Any]]:
    """Load benchmark results from JSON."""
    path = BENCHMARK_DIR / f"{name}_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_hnsw_graph(hnsw_index: Any, name: str = "hnsw") -> None:
    """Serialize the HNSW graph structure to JSON."""
    ensure_dirs()
    path = GENERATED_DIR / f"{name}_graph.json"

    graph_data = {
        "M": hnsw_index.M,
        "ef_construction": hnsw_index.ef_construction,
        "ef_search": hnsw_index.ef_search,
        "max_layer": hnsw_index._max_layer,
        "entry_point": hnsw_index._entry_point,
        "nodes": {},
    }

    for id_, node in hnsw_index._nodes.items():
        graph_data["nodes"][id_] = {
            "deleted": node.deleted,
            "neighbors": {str(k): v for k, v in node.neighbors.items()},
            "metadata": {
                k: v for k, v in node.metadata.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False)
    logger.info(f"Saved HNSW graph ({len(graph_data['nodes'])} nodes) to {path}")
