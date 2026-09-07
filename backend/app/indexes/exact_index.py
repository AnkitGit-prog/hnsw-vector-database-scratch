"""
exact_index.py — Exact Brute-Force Vector Index

This is the GROUND TRUTH index.

Algorithm:
  For every search query, compute cosine similarity against ALL stored
  vectors using fully vectorized NumPy matrix operations, then return the
  top-k highest-similarity results.

Why brute-force?
  Brute-force search is O(N * D) per query (N = number of vectors,
  D = dimension).  It is exact — it will always find the true nearest
  neighbors — but it does not scale well.  For N = 50,000 and D = 384,
  each query requires ~19 million multiply-add operations.

Why NumPy vectorization?
  A Python loop over 50,000 vectors would be ~100× slower than a single
  NumPy matrix–vector multiplication.  We store all vectors as a 2-D
  NumPy matrix (shape N×D) and compute all similarities in one call:

      similarities = matrix @ query_vector    (dot products)

  Because vectors are pre-normalized to unit length, the dot product IS
  the cosine similarity:

      cosine(A, B) = dot(A, B) / (||A|| * ||B||) = dot(A, B)   [if ||A||=||B||=1]

  This single matrix operation runs in highly optimized BLAS and is
  orders of magnitude faster than equivalent Python loops.

This class is intentionally kept simple — no fancy data structures.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExactVectorIndex:
    """
    Exact brute-force vector index using cosine similarity.

    Stores vectors in a dense NumPy matrix.  Search is O(N) exact.
    All stored vectors are L2-normalized on insertion so cosine similarity
    reduces to a single dot-product matrix multiplication.

    Thread safety: a read-write lock is used so multiple threads can
    search concurrently while insert/delete are exclusive.
    """

    def __init__(self) -> None:
        # _ids[i]       → vector ID at row i
        # _matrix       → shape (N, D) float32, rows are L2-normalized
        # _metadata[id] → user metadata dict
        # _deleted       → set of IDs marked as deleted
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None   # built lazily
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._deleted: set = set()
        self._id_to_row: Dict[str, int] = {}        # id → row index (fast lookup)
        self._lock = threading.RLock()

        # Bookkeeping for stats
        self._insert_count = 0
        self._dimension: Optional[int] = None
        self._build_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Return the L2-normalized version of vec.  Safe against zero vectors."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Cannot normalize a zero vector.")
        return vec / norm

    def _rebuild_matrix(self) -> None:
        """
        Rebuild the dense matrix from the current ID list.

        Called after deletions compact the ID list.  In practice, deletions
        are logical (tombstones), so the matrix is NOT rebuilt on every
        delete — only the tombstone set is updated.  The matrix is rebuilt
        only when explicitly requested (e.g., after bulk deletions).
        """
        if not self._ids:
            self._matrix = None
            return
        # Collect rows for active IDs only
        rows = []
        valid_ids = []
        for id_ in self._ids:
            if id_ in self._metadata:  # safety check
                rows.append(self._metadata[id_]["_normalized_vector"])
                valid_ids.append(id_)
        self._ids = valid_ids
        self._id_to_row = {id_: i for i, id_ in enumerate(valid_ids)}
        self._matrix = np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, id_: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Insert a vector with associated metadata.

        Parameters
        ----------
        id_      : unique string identifier
        vector   : raw embedding (will be L2-normalized internally)
        metadata : arbitrary dict — stored as-is alongside the vector
        """
        vec = np.array(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError("Vector contains NaN or Inf values.")
        if len(vec) == 0:
            raise ValueError("Vector must not be empty.")

        with self._lock:
            if self._dimension is None:
                self._dimension = len(vec)
            elif len(vec) != self._dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self._dimension}, got {len(vec)}"
                )

            if id_ in self._id_to_row:
                raise ValueError(f"Duplicate ID: {id_!r} already exists.")

            normalized = self._normalize(vec)

            # Append to dense matrix
            row_idx = len(self._ids)
            self._ids.append(id_)
            self._id_to_row[id_] = row_idx

            if self._matrix is None:
                self._matrix = normalized.reshape(1, -1)
            else:
                self._matrix = np.vstack([self._matrix, normalized])

            # Store metadata + keep normalized vector for rebuild
            meta_copy = dict(metadata)
            meta_copy["_normalized_vector"] = normalized
            self._metadata[id_] = meta_copy
            self._insert_count += 1

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Exact cosine similarity search against ALL active vectors.

        Returns a list of up to top_k dicts:
          {id, similarity, metadata}

        Algorithm:
          1. Normalize query vector.
          2. Compute matrix @ query = dot products (= cosine similarities
             because all rows are already unit-length).
          3. Zero out deleted rows.
          4. Use np.argpartition for efficient top-k without full sort.
          5. Sort only the top-k candidates.
        """
        if self._matrix is None:
            return []
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        qvec = np.array(query_vector, dtype=np.float32)
        if not np.isfinite(qvec).all():
            raise ValueError("Query vector contains NaN or Inf.")
        if len(qvec) != self._dimension:
            raise ValueError(
                f"Query dimension {len(qvec)} != index dimension {self._dimension}"
            )

        with self._lock:
            qvec = self._normalize(qvec)

            # --- vectorized cosine similarity (single BLAS call) ---
            # similarities shape: (N,)
            similarities = self._matrix @ qvec

            # Mask deleted rows with -inf so they never appear in results
            for id_ in self._deleted:
                row = self._id_to_row.get(id_)
                if row is not None:
                    similarities[row] = -np.inf

            n_active = len(self._ids) - len(self._deleted)
            k = min(top_k, n_active)
            if k == 0:
                return []

            # Efficient top-k: argpartition is O(N), much faster than full argsort
            if k < len(similarities):
                top_idx = np.argpartition(similarities, -k)[-k:]
            else:
                top_idx = np.arange(len(similarities))

            # Sort the k candidates by descending similarity
            top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]

            results = []
            for idx in top_idx:
                id_ = self._ids[idx]
                if id_ in self._deleted:
                    continue
                meta = dict(self._metadata[id_])
                meta.pop("_normalized_vector", None)  # don't expose internal field
                results.append(
                    {
                        "id": id_,
                        "similarity": float(similarities[idx]),
                        "metadata": meta,
                    }
                )
            return results

    def delete(self, id_: str) -> bool:
        """
        Logically delete a vector by ID.

        The vector remains in the matrix but is excluded from all search results.
        Returns True if the ID was found and deleted, False if not found.
        """
        with self._lock:
            if id_ not in self._id_to_row:
                return False
            self._deleted.add(id_)
            return True

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a given ID, or None if not found."""
        with self._lock:
            meta = self._metadata.get(id_)
            if meta is None:
                return None
            result = dict(meta)
            result.pop("_normalized_vector", None)
            result["deleted"] = id_ in self._deleted
            return result

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the index."""
        with self._lock:
            total = len(self._ids)
            deleted = len(self._deleted)
            return {
                "type": "ExactBruteForce",
                "total_vectors": total,
                "active_vectors": total - deleted,
                "deleted_vectors": deleted,
                "dimension": self._dimension,
                "build_time_seconds": self._build_time,
                "insert_count": self._insert_count,
            }

    def get_all_ids(self) -> List[str]:
        """Return all active (non-deleted) IDs."""
        with self._lock:
            return [id_ for id_ in self._ids if id_ not in self._deleted]

    def get_normalized_vector(self, id_: str) -> np.ndarray:
        """Return the stored L2-normalized vector for a given ID."""
        with self._lock:
            meta = self._metadata.get(id_)
            if meta is None:
                raise KeyError(f"ID {id_!r} not found.")
            return meta["_normalized_vector"].copy()

    def __len__(self) -> int:
        return len(self._ids) - len(self._deleted)

    def __repr__(self) -> str:
        return (
            f"ExactVectorIndex(vectors={len(self._ids)}, "
            f"deleted={len(self._deleted)}, dim={self._dimension})"
        )
