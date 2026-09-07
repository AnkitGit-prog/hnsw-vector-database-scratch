# INTERVIEW_NOTES.md — Technical Q&A for the Vector Database Project

This file contains detailed answers to expected interview questions about the project.

---

## What is a vector database?

A vector database is a storage and retrieval system designed for dense floating-point vectors called *embeddings*. Unlike relational databases that answer exact queries ("find all users with age > 30"), a vector database answers *similarity queries*: "find the K stored vectors most similar to this query vector."

The similarity metric is typically cosine similarity or Euclidean distance. Vector databases power semantic search, recommendation engines, image similarity search, and RAG (Retrieval-Augmented Generation) systems for LLMs.

---

## Why are embeddings required?

Raw text, images, and audio cannot be directly compared for semantic similarity — they are high-dimensional, discrete, and unstructured. *Embeddings* are dense, continuous vector representations learned by neural networks so that semantically similar inputs are close together in vector space.

Example: "dog" and "puppy" may have very different token IDs but highly similar embedding vectors because they co-occur in similar contexts during training.

Without embeddings, you cannot measure semantic similarity — you can only do keyword matching.

---

## What is cosine similarity?

Cosine similarity measures the *angle* between two vectors, independent of their magnitude:

```
cosine(A, B) = dot(A, B) / (||A|| × ||B||)
```

Range: -1 (opposite) to +1 (identical direction).

**Why cosine, not Euclidean?**
For embeddings, the magnitude of a vector reflects the *confidence* of the encoder, not the meaning. Two vectors can represent the same concept at different magnitudes. Cosine similarity measures semantic direction, not magnitude.

**Key optimization:** If vectors are pre-normalized to unit length, `cosine(A, B) = dot(A, B)`. This allows computing 50,000 similarities with a single matrix-vector multiply in NumPy.

---

## Why is brute-force exact?

The brute-force index computes the similarity between the query and **every single stored vector**. Since no candidate is skipped, the result is mathematically guaranteed to be the true nearest neighbors.

It is the gold standard — the *ground truth* against which all approximate algorithms are measured.

---

## What is ANN?

Approximate Nearest-Neighbor (ANN) search finds vectors that are *very likely* to be the nearest neighbors, but with no absolute guarantee. The approximation is controlled by parameters: higher quality parameters give results closer to the true nearest neighbors, at the cost of more computation.

In practice, ANN with recall@10 = 0.95 means 9.5 of the 10 returned results are in the true top-10. For most applications, this is indistinguishable from exact search.

---

## Why is brute-force O(N)?

For each query, brute-force performs N dot-product comparisons (one per stored vector). Each dot-product is O(D) operations. Total: O(N × D) per query.

For N = 50,000 and D = 384: ~19.2 million multiply-adds per query. At 1 billion vectors: 384 billion multiply-adds per query — clearly infeasible for real-time use.

---

## What is HNSW?

HNSW stands for **Hierarchical Navigable Small World**. It is a graph-based ANN algorithm published by Malkov and Yashunin (2016/2018). It builds a multilayer graph where:
- Higher layers have sparse, long-range connections ("highways")
- Layer 0 has dense, short-range connections
- Each node's maximum layer is drawn from a geometric distribution

Search traverses the graph greedily from the top layer, narrowing in on the query's neighborhood, then performs a wider beam search at layer 0.

---

## Why does HNSW use multiple layers?

A single-layer navigable small-world graph suffers from a problem: greedy search can get stuck at local minima far from the true nearest neighbors. The logarithmic routing problem requires O(log²N) steps.

By adding hierarchical layers — where higher layers have fewer nodes and longer-range edges — the initial greedy descent rapidly approaches the query's neighborhood before reaching the dense layer 0. This reduces search complexity to O(log N) empirically.

The principle is analogous to highway navigation: use motorways (high layers) to get to the right city, then use local roads (layer 0) to find the exact destination.

---

## What is a small-world graph?

A small-world graph has two properties:
1. **High local clustering** — neighbors of a node tend to be connected to each other
2. **Short average path length** — any two nodes are connected by a small number of hops (the "six degrees of separation" phenomenon)

In a small-world graph, greedy routing (always move to the neighbor closest to the destination) is very efficient. HNSW constructs a navigable small-world graph with the additional property that greedy routing is *provably* efficient.

---

## What is M?

M is the maximum number of neighbors each node can have per layer (except layer 0, which allows 2M).

**Effect of M:**
- Higher M → more connections → better recall → more memory → slower insertion
- Lower M → fewer connections → lower recall → less memory → faster insertion

Typical values: M = 8 (fast, less memory), M = 16 (balanced, default), M = 32 (high recall, large datasets).

Memory scales as O(N × M × max_layer).

---

## What is ef_construction?

ef_construction is the beam width used during *insertion*. It controls the quality of the graph structure:

- Higher ef_construction → more candidates explored when selecting neighbors → better graph quality → higher recall at search time → slower insertions
- Lower ef_construction → faster insertions → lower graph quality → lower recall

ef_construction only affects the index build phase — it has no effect on search speed after the index is built.

Typical values: 100–400.

---

## What is ef_search?

ef_search is the beam width used during *search*. It is the primary accuracy–latency knob:

- Higher ef_search → more candidates explored at layer 0 → higher recall → slower search
- Lower ef_search → fewer candidates → lower recall → faster search

ef_search can be changed dynamically without rebuilding the index. This is the key parameter for tuning the accuracy–latency tradeoff in production.

Typical values: 10 (fast, ~80% recall) to 200 (slow, ~99% recall).

---

## How does HNSW insertion work?

1. **Draw level:** `l = floor(-ln(U(0,1)) × mL)` where mL = 1/ln(M)
2. **Greedy descent:** Starting from the global entry point at `max_layer`, greedily descend to layer `l+1`, tracking the single nearest neighbor at each layer.
3. **Layer-by-layer insertion:** For each layer from `min(l, max_layer)` down to 0:
   - Run beam search with `ef_construction` candidates
   - Select best M neighbors
   - Wire bidirectional edges between the new node and the M neighbors
   - Prune neighbor lists that exceed M_max (degree bound)
4. **Update entry point:** If `l > max_layer`, the new node becomes the global entry point.

---

## How does HNSW search work?

1. **Start:** Initialize candidate set with the global entry point.
2. **High-layer descent (layers max_layer → 1):** At each layer, find the single nearest neighbor among the current candidates' neighborhood. Move to that neighbor. Continue until no improvement.
3. **Layer 0 beam search:** Expand to ef_search candidates. Maintain a result heap and a candidate heap. At each step, pop the best unvisited candidate, explore its layer-0 neighbors, update both heaps.
4. **Return:** Top-k results from the layer-0 result set, excluding deleted nodes.

The key invariant: the candidate heap always contains nodes worth exploring; the result heap always contains the best nodes found so far.

---

## Why is HNSW approximate?

1. **Greedy descent can miss:** At high layers, the greedy step always moves to the *current best* neighbor, which may not be the globally best path to the query.
2. **Limited neighborhood exploration:** ef_search limits how many candidates are explored at layer 0. With ef_search = 50, at most 50 candidates are evaluated, even if more relevant nodes exist nearby.
3. **Graph construction quality:** If ef_construction was low during build, the graph may have poor connections, leading to worse search results.

The approximation is controlled but not eliminated. This is the fundamental tradeoff of ANN.

---

## What happens if ef_search increases?

- More candidates are explored at layer 0.
- Higher probability of finding all true nearest neighbors.
- Recall@10 increases (approaches 1.0 asymptotically).
- Search latency increases (more nodes visited per query).
- Speedup over exact search decreases.

At extreme ef_search values (ef_search = N), HNSW degenerates toward an exhaustive search and recall approaches 1.0. At ef_search = 1, it is nearly a pure greedy search with low recall.

---

## Why does recall increase?

Recall increases because with a larger ef_search budget, the beam search can explore more of the layer-0 neighborhood before declaring the result. More exploration → higher probability that all 10 true nearest neighbors fall within the explored region.

---

## Why does latency increase?

Each candidate in the beam requires: fetching the node's neighbors (dict lookup), computing cosine similarity for each neighbor (dot product), and heap operations. More candidates = more work = higher latency.

The relationship is approximately linear: doubling ef_search roughly doubles search latency for dense, uniform datasets.

---

## How is recall@10 calculated?

```python
recall_at_10 = len(set(exact_top_10) ∩ set(hnsw_top_10)) / 10
```

1. Run ExactVectorIndex.search(query, 10) → get exact_top_10 IDs (ground truth)
2. Run HNSWIndex.search(query, 10) → get hnsw_top_10 IDs
3. Count how many hnsw IDs appear in the exact set
4. Divide by 10

Averaged over 500 benchmark queries: `mean_recall_at_10`.

---

## Why do we need a brute-force ground truth?

Without ground truth, there is no way to measure accuracy. The "approximate" label means nothing unless you know what the exact answer is.

The exact brute-force index provides this ground truth:
- It is mathematically provable to be correct.
- Every ANN algorithm is measured against it.
- Without it, recall is unmeasurable.

This is why the ExactVectorIndex is the most important component — not for production use, but as the correctness oracle.

---

## How does deletion work in HNSW?

**Tombstone (logical) deletion** (used in this implementation):
1. Set `node.deleted = True`.
2. Add the node's ID to the `_deleted` set.
3. Exclude deleted nodes from search results.
4. Keep all graph edges intact.

**Why not physical deletion?**
Removing a node from an HNSW graph requires:
- Finding all nodes that have this node as a neighbor (reverse edge lookup — O(N))
- Removing those edges
- Potentially reconnecting those neighbors to maintain the degree ≥ M_min invariant
- Special handling if the deleted node is the entry point
- Risk of disconnecting the graph if the node was a bridge

This is much more complex than insertion and can cascade into multiple graph repair operations. Production systems (hnswlib, Weaviate) use tombstones and periodic rebuilds for the same reason.

**Periodic rebuild:** Call `HNSWIndex.rebuild()` to physically remove all tombstones and rebuild the graph from the active nodes. This restores optimal graph quality.

---

## What are HNSW's limitations?

1. **Memory:** O(N × M × layers) edges + O(N × D) vectors. Scales poorly with very large N.
2. **Insert-only:** Adding new nodes is efficient; removing nodes is complex (tombstone workaround).
3. **No updates:** Changing a vector requires delete + insert.
4. **Single-threaded Python:** This implementation is not thread-safe for concurrent inserts.
5. **No filtering:** Cannot efficiently filter by metadata during search.
6. **Recall depends on graph quality:** A poorly built graph (low ef_construction) can never achieve high recall regardless of ef_search.
7. **Cold start:** The first few nodes inserted create a graph with poor connectivity; quality improves as more nodes are added.

---

## What happens with 1 million vectors?

**Memory:** 1M × 384 × 4 bytes ≈ 1.5 GB for vectors. Plus HNSW edges: 1M × 16 × 2 (bidirectional) × 8 bytes (pointer) ≈ 256 MB. Total: ~2 GB — feasible on a modern server.

**Build time:** HNSW insertion is O(M × ef_construction × log N) per node. For 1M nodes: roughly 10–30 minutes on CPU (reference hnswlib implementation: ~2 min with C++).

**Search latency:** Increases slowly with N (O(log N) empirically). For 1M vectors, expect HNSW search to be 2–3× slower than for 50K vectors.

**Exact search:** For 1M × 384 vectors, brute-force would take ~500ms per query — clearly unusable for real-time applications. This is precisely why HNSW is needed.

---

## How would you improve the system for production?

1. **C++ or Rust core** — Python HNSW is educational; production needs hnswlib/usearch speed.
2. **Concurrent reads with exclusive writes** — use proper `rwlock` rather than Python's `RLock`.
3. **Memory-mapped vectors** — use `numpy.memmap` to avoid loading all vectors into RAM.
4. **Vector quantization** — reduce memory 8-32× with Product Quantization, accepting minor recall loss.
5. **Batch inserts** — accept bulk inserts to amortize per-insert overhead.
6. **WAL (Write-Ahead Log)** — persist inserts to disk before acknowledging, for crash safety.
7. **Tiered storage** — keep hot vectors in RAM, cold vectors on SSD.

---

## How would you persist the index?

**Vectors:** Already saved as NumPy `.npz` — efficient binary format, fast load/save.

**HNSW graph:**
- Serialize `_nodes` adjacency lists as JSON (simple, human-readable) — already implemented
- Binary format (custom or Protocol Buffers) for faster load/save at scale
- Or: on restart, rebuild the HNSW from scratch from the saved vectors (simpler, takes a few minutes)

**Metadata:** JSON file per-document, or SQLite for indexed metadata queries.

---

## How would you support filtering?

**Pre-filtering:** Filter metadata (e.g., `category == "technology"`) before searching → only search the filtered subset. Simple but requires an extra data structure (inverted index on metadata fields).

**Post-filtering:** Search HNSW → filter results → if too few pass the filter, expand ef_search and repeat. Works well when filter selectivity is low.

**Filtered graph (ACORN/Weaviate approach):** Precompute per-category HNSW subgraphs. High quality but requires more memory and rebuilds when categories change.

---

## How would you distribute the vector database?

**Horizontal sharding:**
1. Partition vectors by ID range, hash, or semantic cluster across N nodes.
2. Each node holds 1/N of the vectors and its own HNSW index.
3. A coordinator sends the query to all shards in parallel.
4. Shards return their local top-k.
5. Coordinator merges and re-ranks the N × top-k candidates.

**Replication:** Each shard replicated across 3 nodes for fault tolerance.

**Challenge:** Recall may drop at shard boundaries. Solution: overlap partitions (each vector stored in 2 shards).

---

## How would you reduce memory usage?

1. **float16 vectors** — halve memory with minimal recall impact (384-dim: 768 bytes → 384 bytes per vector)
2. **int8 quantization** — 75% reduction, slight recall loss
3. **Product Quantization (PQ)** — encode each 384-dim vector as 48 × int8 codes → 96% reduction; requires distance table lookup instead of dot product
4. **Scalar quantization (SQ8)** — per-dimension 8-bit quantization, 75% reduction, very fast
5. **Memory-mapped files** — vectors on SSD, only active vectors in RAM
6. **Eviction policy** — LRU cache for frequently queried vectors, evict cold vectors to disk
