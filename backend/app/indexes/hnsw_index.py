"""
hnsw_index.py — Hierarchical Navigable Small World (HNSW) Index

HNSW is an approximate nearest-neighbor (ANN) algorithm that organizes
vectors into a multilayer graph to achieve sublinear search time in practice.

===============================================================================
ALGORITHM OVERVIEW
===============================================================================

The key insight is to build a "navigable small-world" graph where:
  - Long-range edges at higher layers enable fast global navigation.
  - Short-range edges at layer 0 provide fine-grained local search.

INSERTION:
  1. Draw the node's maximum layer `l` from a geometric distribution:
       P(l >= k) = (1/M)^k   where M is the number of neighbors per layer.
     This gives the famous HNSW pyramid shape.
  2. Start from the global entry point at the current maximum layer.
  3. Descend greedily from top layer to l+1, moving only the 1-NN at each step.
  4. From layer min(l, max_layer) down to 0:
       - Run the ef_construction beam search to find the best ef_construction
         candidates in that layer's graph.
       - Select up to M best candidates as neighbors.
       - Connect the new node to those neighbors (bidirectional edges).
       - Apply neighbor pruning (heuristic) to keep each neighbor list ≤ M.
  5. If the new node's level l > current max_layer, update the entry point.

SEARCH:
  1. Start at the global entry point at the current highest layer.
  2. Greedily descend from top to layer 1:
       - At each layer, move to the neighbor closest to the query.
       - Continue until no neighbor is closer (local minimum).
  3. At layer 0, run a wider beam search with ef_search candidates.
  4. Return the top-k results from the layer-0 candidate set.

WHY IS HNSW APPROXIMATE?
  The greedy descent can get stuck in a local minimum at higher layers and
  enter layer 0 from a suboptimal starting point.  The wider ef_search
  mitigates this but cannot guarantee finding the global optimum.

DELETION (TOMBSTONE STRATEGY):
  Physical removal of a node from an HNSW graph is complex because:
    - The node may be the entry point (special handling required).
    - Removing a node's edges can disconnect graph regions.
    - Its former neighbors may now have fewer connections than required,
      breaking the small-world property.
  We use LOGICAL (tombstone) deletion:
    - Mark the node deleted=True.
    - Exclude it from all search results.
    - Keep its edges intact to preserve graph connectivity.
  This is the same strategy used by hnswlib and many production systems.
  A periodic maintenance/rebuild operation can physically remove tombstones.

PARAMETERS:
  M              — max neighbors per node per layer (default 16)
                   Higher M → better recall, more memory, slower insert.
  ef_construction — beam width during insertion (default 200)
                   Higher → better graph quality, slower inserts.
  ef_search      — beam width during search (default 50)
                   Higher → better recall, slower search.
  mL             — level normalization factor = 1 / ln(M)
                   Controls how often higher layers are assigned.

COMPLEXITY:
  Build:  O(N * M * ef_construction * log N)  approximate
  Search: O(log N * M) per query  (empirically sublinear)
  Memory: O(N * M * max_layers)

REFERENCES:
  Malkov & Yashunin, 2018. "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs."
  https://arxiv.org/abs/1603.09320
===============================================================================
"""

from __future__ import annotations

import heapq
import math
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .node import HNSWNode


# ---------------------------------------------------------------------------
# Type alias: a "heap entry" is (neg_similarity, id)
# We negate similarity because Python's heapq is a min-heap.
# ---------------------------------------------------------------------------
HeapEntry = Tuple[float, str]


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) approximate vector index.

    All vector arithmetic uses NumPy.  The graph structure (adjacency lists)
    uses plain Python dicts and lists — intentionally simple and inspectable.

    Parameters
    ----------
    M              : int   — max neighbors per layer (default 16)
    ef_construction: int   — beam width during insert (default 200)
    ef_search      : int   — default beam width during search (default 50)
    seed           : int   — random seed for reproducible level assignment
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        # ---- configurable parameters ----
        self.M = M
        self.M_max0 = M * 2       # layer-0 allows more neighbors (standard HNSW)
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Level normalization factor from the paper: mL = 1 / ln(M)
        # Controls the geometric distribution of levels.
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0

        # ---- graph state ----
        self._nodes: Dict[str, HNSWNode] = {}    # id → node
        self._entry_point: Optional[str] = None  # current global entry point ID
        self._max_layer: int = -1                # current highest layer index
        self._dimension: Optional[int] = None

        # ---- tombstone set for fast membership test ----
        self._deleted: Set[str] = set()

        # ---- thread safety ----
        self._lock = threading.RLock()

        # ---- reproducible level assignment ----
        self._rng = random.Random(seed)

        # ---- bookkeeping ----
        self._insert_count = 0
        self._build_time: float = 0.0

    # ======================================================================
    # Similarity
    # ======================================================================

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector.  Raises on zero vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Cannot normalize a zero vector.")
        return vec / norm

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalized vectors.
        Since both are unit vectors: cosine(a, b) = dot(a, b).
        Clipped to [-1, 1] to guard against floating-point drift.
        """
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    # ======================================================================
    # Level assignment
    # ======================================================================

    def _random_level(self) -> int:
        """
        Sample the layer index for a new node.

        From the HNSW paper:
          level = floor(-ln(uniform(0,1)) * mL)

        This yields a geometric distribution:
          P(level >= k) ≈ exp(-k / mL) = (1/M)^k

        Most nodes get level 0.  Fewer get level 1, even fewer level 2, etc.
        This creates the pyramid structure that enables fast navigation.
        """
        level = 0
        while self._rng.random() < (1.0 / self.M):
            level += 1
        return level

    # ======================================================================
    # Core graph traversal helpers
    # ======================================================================

    def _get_vector(self, id_: str) -> np.ndarray:
        """Return the stored (normalized) vector for a node ID."""
        return self._nodes[id_].vector

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: List[str],
        ef: int,
        layer: int,
        exclude_deleted: bool = False,
    ) -> List[HeapEntry]:
        """
        Beam search within a single HNSW layer.

        This is the core inner loop of HNSW — the same routine is used
        for both insertion (ef = ef_construction) and search (ef = ef_search).

        Algorithm:
          Maintain two heaps:
            candidates  : min-heap of (-sim, id) — next nodes to explore
            results     : max-heap of (-sim, id) — best ef results found so far
          visited set prevents revisiting nodes.

          While candidates is non-empty:
            1. Pop the best (highest similarity) candidate c.
            2. Compare c's similarity to the worst result in results.
               If c is worse than the worst result AND |results| >= ef → stop.
            3. For each unvisited neighbor n of c at this layer:
               a. Compute sim(query, n).
               b. If n is better than worst result OR |results| < ef:
                  - Push n onto candidates.
                  - Push n onto results; trim results to ef if overfull.

        Returns list of HeapEntry sorted by descending similarity (best first).

        Parameters
        ----------
        query        : normalized query vector
        entry_ids    : starting node IDs (already explored)
        ef           : beam width — maximum number of candidates to return
        layer        : which layer's neighbor lists to use
        exclude_deleted: if True, skip deleted nodes in neighbor traversal
        """
        visited: Set[str] = set()
        # candidates: min-heap of (neg_sim, id) — ordered best-first for greedy pop
        candidates: List[HeapEntry] = []
        # results: max-heap of (pos_sim, id) so heappop removes the WORST result
        # We store (+sim, id) instead of (-sim, id) for the results heap.
        results_maxheap: List[tuple] = []  # (sim, id), max-heap via negation trick

        for eid in entry_ids:
            if eid not in self._nodes:
                continue
            sim = self._cosine(query, self._get_vector(eid))
            heapq.heappush(candidates, (-sim, eid))
            heapq.heappush(results_maxheap, (sim, eid))   # max-heap: larger sim = better
            visited.add(eid)

        while candidates:
            # Pop the candidate with HIGHEST similarity (lowest neg_sim)
            c_neg_sim, c_id = heapq.heappop(candidates)
            c_sim = -c_neg_sim

            # Worst result in results: smallest sim in the max-heap = results_maxheap[0]
            if results_maxheap:
                worst_sim = results_maxheap[0][0]
            else:
                worst_sim = -float("inf")

            # Pruning: if current candidate is worse than all results, stop.
            if c_sim < worst_sim and len(results_maxheap) >= ef:
                break

            # Explore neighbors
            node = self._nodes.get(c_id)
            if node is None:
                continue

            for neighbor_id in node.get_neighbors(layer):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor_node = self._nodes.get(neighbor_id)
                if neighbor_node is None:
                    continue

                if exclude_deleted and neighbor_node.deleted:
                    continue

                n_sim = self._cosine(query, self._get_vector(neighbor_id))

                # Accept if results not full OR neighbor is better than worst result
                worst_sim_now = results_maxheap[0][0] if results_maxheap else -float("inf")
                if len(results_maxheap) < ef or n_sim > worst_sim_now:
                    heapq.heappush(candidates, (-n_sim, neighbor_id))
                    heapq.heappush(results_maxheap, (n_sim, neighbor_id))

                    # Trim results: remove the WORST result (smallest sim = heap root)
                    if len(results_maxheap) > ef:
                        heapq.heappop(results_maxheap)  # removes minimum sim = worst

        # Convert results_maxheap to list sorted by descending similarity (best first)
        # results_maxheap is a min-heap on sim, so we extract all and sort descending.
        all_results = sorted(results_maxheap, reverse=True)  # best sim first
        # Convert to (neg_sim, id) format for consistency with callers
        return [(-sim, id_) for sim, id_ in all_results]

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[HeapEntry], M: int
    ) -> List[str]:
        """
        Select the best M neighbors from a candidate list.

        Simple strategy: take the M candidates with highest similarity.
        The HNSW paper also describes a heuristic that prefers diverse
        neighbors; we use the simple version for clarity.

        Parameters
        ----------
        query      : normalized query vector (the new node being inserted)
        candidates : list of (neg_sim, id) from _search_layer
        M          : maximum neighbors to select

        Returns list of up to M neighbor IDs.
        """
        # candidates is sorted ascending by neg_sim → first entry is best (lowest neg_sim)
        selected = []
        for neg_sim, id_ in candidates:
            if len(selected) >= M:
                break
            selected.append(id_)
        return selected

    def _prune_neighbors(
        self, node_id: str, layer: int, M_max: int
    ) -> None:
        """
        Trim a node's neighbor list to at most M_max entries.

        When a node gains a new neighbor that pushes it over M_max, we
        keep only the M_max nearest neighbors (by cosine similarity).
        This maintains the degree bound that makes HNSW efficient.
        """
        node = self._nodes[node_id]
        neighbors = node.get_neighbors(layer)
        if len(neighbors) <= M_max:
            return

        # Compute similarity to all current neighbors
        node_vec = node.vector
        scored = [
            (-self._cosine(node_vec, self._nodes[nid].vector), nid)
            for nid in neighbors
            if nid in self._nodes
        ]
        # Keep the M_max closest
        scored.sort()
        node.set_neighbors(layer, [nid for _, nid in scored[:M_max]])

    # ======================================================================
    # Public API — Insert
    # ======================================================================

    def insert(self, id_: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Insert a new vector into the HNSW graph.

        Steps:
          1. Validate and normalize vector.
          2. Create HNSWNode.
          3. Draw a random level l.
          4. Greedy descent from max_layer down to l+1 (single NN per step).
          5. For layers l down to 0:
               beam search with ef_construction → select neighbors → wire edges.
          6. Update entry point if l > max_layer.

        Parameters
        ----------
        id_      : unique string identifier
        vector   : raw embedding vector (will be normalized)
        metadata : arbitrary dict stored alongside the vector
        """
        vec = np.array(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {vec.shape}")
        if len(vec) == 0:
            raise ValueError("Vector must not be empty.")
        if not np.isfinite(vec).all():
            raise ValueError("Vector contains NaN or Inf.")

        with self._lock:
            if self._dimension is None:
                self._dimension = len(vec)
            elif len(vec) != self._dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self._dimension}, got {len(vec)}"
                )

            if id_ in self._nodes:
                raise ValueError(f"Duplicate ID: {id_!r} already exists.")

            # Normalize once; stored vector is always unit-length
            normalized = self._normalize(vec)

            # --- Step 1: Create node ---
            node = HNSWNode(id=id_, vector=normalized, metadata=dict(metadata))
            self._nodes[id_] = node

            # --- Step 2: Draw random level ---
            l = self._random_level()

            # If this is the first node, make it the entry point at its level
            if self._entry_point is None:
                for layer in range(l + 1):
                    node.set_neighbors(layer, [])
                self._entry_point = id_
                self._max_layer = l
                self._insert_count += 1
                return

            # --- Step 3: Greedy descent from top to l+1 ---
            # At layers above the new node's level, we only need the single
            # nearest neighbor as a stepping stone — no need for wider search.
            entry_ids = [self._entry_point]
            current_max = self._max_layer

            for layer in range(current_max, l, -1):
                # Single-NN greedy descent (ef=1)
                layer_results = self._search_layer(normalized, entry_ids, ef=1, layer=layer)
                if layer_results:
                    entry_ids = [layer_results[0][1]]

            # --- Step 4: For each layer from min(l, max_layer) down to 0 ---
            for layer in range(min(l, current_max), -1, -1):
                M_max = self.M_max0 if layer == 0 else self.M
                ef = max(self.ef_construction, M_max)

                # Beam search to find ef_construction nearest candidates
                candidates = self._search_layer(normalized, entry_ids, ef=ef, layer=layer)

                # Select best M neighbors
                selected_ids = self._select_neighbors(normalized, candidates, M_max)

                # Wire edges: new node → selected neighbors
                node.set_neighbors(layer, selected_ids)

                # Wire edges: selected neighbors → new node (bidirectional)
                for neighbor_id in selected_ids:
                    neighbor_node = self._nodes[neighbor_id]
                    neighbor_node.add_neighbor(layer, id_)
                    # Prune neighbor's list if it exceeded M_max
                    self._prune_neighbors(neighbor_id, layer, M_max)

                # Use these candidates as entry points for the next (lower) layer
                entry_ids = [eid for _, eid in candidates[:ef]]

            # --- Step 5: Update entry point if new node is at a higher level ---
            if l > self._max_layer:
                self._entry_point = id_
                self._max_layer = l
                # Initialize empty neighbor lists for new layers
                for new_layer in range(current_max + 1, l + 1):
                    node.set_neighbors(new_layer, [])

            self._insert_count += 1

    # ======================================================================
    # Public API — Search
    # ======================================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Approximate nearest-neighbor search using HNSW graph traversal.

        This does NOT fall back to brute-force.  All results come from
        graph traversal only.

        Algorithm:
          1. Normalize query vector.
          2. Greedy descent from max_layer down to layer 1 (single NN per step).
          3. At layer 0, beam search with ef_search candidates.
          4. Exclude deleted nodes from results.
          5. Return top-k by cosine similarity.

        Parameters
        ----------
        query_vector : raw query embedding
        top_k        : number of results to return
        ef_search    : beam width override (default: self.ef_search)

        Returns list of dicts: {id, similarity, metadata}
        """
        if self._entry_point is None:
            return []
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, top_k)  # ef must be at least top_k

        qvec = np.array(query_vector, dtype=np.float32)
        if not np.isfinite(qvec).all():
            raise ValueError("Query vector contains NaN or Inf.")
        if len(qvec) != self._dimension:
            raise ValueError(
                f"Query dimension {len(qvec)} != index dimension {self._dimension}"
            )

        with self._lock:
            qvec = self._normalize(qvec)

            # --- Greedy descent from top layer to layer 1 ---
            entry_ids = [self._entry_point]
            for layer in range(self._max_layer, 0, -1):
                layer_results = self._search_layer(
                    qvec, entry_ids, ef=1, layer=layer, exclude_deleted=True
                )
                if layer_results:
                    entry_ids = [layer_results[0][1]]

            # --- Wide beam search at layer 0 ---
            candidates = self._search_layer(
                qvec, entry_ids, ef=ef, layer=0, exclude_deleted=True
            )

            # Collect top-k non-deleted results
            results = []
            for neg_sim, id_ in candidates:
                if len(results) >= top_k:
                    break
                node = self._nodes.get(id_)
                if node is None or node.deleted:
                    continue
                meta = dict(node.metadata)
                results.append(
                    {
                        "id": id_,
                        "similarity": float(-neg_sim),
                        "metadata": meta,
                    }
                )

            return results

    # ======================================================================
    # Public API — Delete
    # ======================================================================

    def delete(self, id_: str) -> bool:
        """
        Logically delete a node using a tombstone.

        WHY NOT PHYSICAL DELETION?
        ─────────────────────────
        Physical deletion from an HNSW graph is significantly more complex
        than insertion for the following reasons:

        1. CONNECTIVITY RISK: A deleted node may be the ONLY bridge between
           two graph regions.  Removing it and its edges can disconnect the
           graph, making some vectors unreachable from the entry point.

        2. ENTRY POINT: If the deleted node is the current global entry
           point, a new valid entry point must be found — non-trivial in a
           multilayer graph.

        3. DEGREE INVARIANT: Neighbors of the deleted node now have one
           fewer connection.  To maintain the M minimum-degree property,
           new replacement edges must be found, which requires running
           the HNSW search algorithm for each affected neighbor.

        4. CASCADE EFFECT: Those reconnection operations can cause further
           structural changes, potentially requiring multiple passes.

        TOMBSTONE STRATEGY (used here):
        ────────────────────────────────
        • Set node.deleted = True.
        • Exclude deleted nodes from search results.
        • Keep all edges intact — graph connectivity is preserved.
        • Periodic rebuild (call rebuild()) physically removes tombstones
          by rebuilding the entire graph from scratch.

        This is the same approach used by hnswlib and is the standard
        recommendation in production vector database systems.

        Returns True if found and marked deleted, False if ID not found.
        """
        with self._lock:
            node = self._nodes.get(id_)
            if node is None:
                return False
            node.deleted = True
            self._deleted.add(id_)
            return True

    def rebuild(self) -> int:
        """
        Physical rebuild: re-insert all non-deleted nodes into a fresh graph.

        This removes tombstones permanently.  Returns number of nodes rebuilt.
        Use this after a large number of deletions to reclaim memory and
        restore optimal graph quality.
        """
        with self._lock:
            active_nodes = [
                n for n in self._nodes.values() if not n.deleted
            ]
            # Re-initialize graph state
            old_M = self.M
            old_ef = self.ef_construction
            old_ef_search = self.ef_search
            self.__init__(M=old_M, ef_construction=old_ef, ef_search=old_ef_search)

            for node in active_nodes:
                self.insert(node.id, node.vector, node.metadata)

            return len(active_nodes)

    # ======================================================================
    # Stats and utilities
    # ======================================================================

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a node, or None if not found."""
        with self._lock:
            node = self._nodes.get(id_)
            if node is None:
                return None
            return {
                "id": node.id,
                "metadata": dict(node.metadata),
                "deleted": node.deleted,
                "max_layer": node.max_layer(),
                "neighbor_counts": {
                    layer: len(nbrs)
                    for layer, nbrs in node.neighbors.items()
                },
            }

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            total = len(self._nodes)
            deleted = len(self._deleted)
            # Count total edges
            total_edges = sum(
                sum(len(nbrs) for nbrs in node.neighbors.values())
                for node in self._nodes.values()
            )
            return {
                "type": "HNSW",
                "total_vectors": total,
                "active_vectors": total - deleted,
                "deleted_vectors": deleted,
                "dimension": self._dimension,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "max_layer": self._max_layer,
                "entry_point": self._entry_point,
                "total_edges": total_edges,
                "build_time_seconds": self._build_time,
                "insert_count": self._insert_count,
            }

    def update_ef_search(self, ef_search: int) -> None:
        """Dynamically update ef_search without rebuilding."""
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}")
        self.ef_search = ef_search

    def __len__(self) -> int:
        return len(self._nodes) - len(self._deleted)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(vectors={len(self._nodes)}, deleted={len(self._deleted)}, "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, max_layer={self._max_layer})"
        )
