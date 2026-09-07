"""
node.py — HNSW Node Data Structure

Each node in the HNSW graph stores:
  - id: unique string identifier
  - vector: numpy float32 array (the embedding)
  - metadata: arbitrary dict (text, source, category, etc.)
  - neighbors: per-layer adjacency list  {layer_idx: [neighbor_id, ...]}
  - deleted: tombstone flag for logical deletion

Why separate neighbors per layer?
  HNSW uses a multilayer graph.  Nodes at higher layers have fewer,
  long-range connections; layer-0 has the densest, short-range connections.
  Keeping separate neighbor lists per layer is the standard implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np


@dataclass
class HNSWNode:
    """A single node in the HNSW graph."""

    id: str
    vector: np.ndarray          # shape (D,), dtype float32, L2-normalized
    metadata: Dict[str, Any]    # arbitrary user metadata

    # neighbors[layer] = list of neighbor node IDs at that layer
    neighbors: Dict[int, List[str]] = field(default_factory=dict)

    # Tombstone flag — set True on delete; node stays in graph but is
    # excluded from search results.
    deleted: bool = False

    def get_neighbors(self, layer: int) -> List[str]:
        """Return neighbor list at the given layer (empty list if none)."""
        return self.neighbors.get(layer, [])

    def set_neighbors(self, layer: int, neighbor_ids: List[str]) -> None:
        """Replace neighbor list at the given layer."""
        self.neighbors[layer] = neighbor_ids

    def add_neighbor(self, layer: int, neighbor_id: str) -> None:
        """Append a neighbor at the given layer (no-op if already present)."""
        if layer not in self.neighbors:
            self.neighbors[layer] = []
        if neighbor_id not in self.neighbors[layer]:
            self.neighbors[layer].append(neighbor_id)

    def max_layer(self) -> int:
        """Highest layer this node participates in."""
        if not self.neighbors:
            return 0
        return max(self.neighbors.keys())

    def __repr__(self) -> str:
        status = "DELETED" if self.deleted else "active"
        return (
            f"HNSWNode(id={self.id!r}, dim={len(self.vector)}, "
            f"layers={sorted(self.neighbors.keys())}, status={status})"
        )
