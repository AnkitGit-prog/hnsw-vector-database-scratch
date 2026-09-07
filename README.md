# Vector Database From Scratch

> A complete vector search engine implementing two indexes from scratch using NumPy:
> an exact brute-force index (ground truth) and a Hierarchical Navigable Small World (HNSW) approximate index.
> No FAISS. No Pinecone. No Chroma. No sklearn.neighbors.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Why Vector Databases?](#2-why-vector-databases)
3. [Why Brute-Force Is Exact But Slow](#3-why-brute-force-is-exact-but-slow)
4. [Why Approximate Nearest-Neighbor Search?](#4-why-approximate-nearest-neighbor-search)
5. [Why HNSW?](#5-why-hnsw)
6. [How HNSW Works](#6-how-hnsw-works)
7. [Architecture](#7-architecture)
8. [Exact Index Implementation](#8-exact-index-implementation)
9. [HNSW Implementation](#9-hnsw-implementation)
10. [Cosine Similarity](#10-cosine-similarity)
11. [Insert, Search, Delete](#11-insert-search-delete)
12. [Dataset Generation](#12-dataset-generation)
13. [Benchmark Methodology](#13-benchmark-methodology)
14. [Recall@10](#14-recall10)
15. [Latency & Speedup](#15-latency--speedup)
16. [Accuracy vs Latency Tradeoff](#16-accuracy-vs-latency-tradeoff)
17. [Limitations](#17-limitations)
18. [How to Run](#18-how-to-run)
19. [API Documentation](#19-api-documentation)
20. [Example Queries](#20-example-queries)
21. [Why I Did Not Use FAISS / Pinecone / Chroma / sklearn.neighbors](#21-why-i-did-not-use-faiss--pinecone--chroma--sklearnneighbors)
22. [Future Improvements](#22-future-improvements)

---

## 1. Project Overview

This project builds a **vector similarity search engine** from scratch.

**What it includes:**
- `ExactVectorIndex` — brute-force cosine similarity over all stored vectors, the ground truth
- `HNSWIndex` — Hierarchical Navigable Small World approximate index, built entirely with NumPy + Python
- **FastAPI** REST backend with insert, search (both types), delete, and benchmark endpoints
- **React + TypeScript** frontend dashboard with 5 pages: semantic search, index overview, benchmark charts, comparison view, insert/delete
- 50,000+ real text embeddings from the 20 Newsgroups corpus
- 500-query benchmark with recall@10, latency, speedup, and parameter sweep

---

## 2. Why Vector Databases?

Traditional databases store structured data and answer exact queries ("find all rows where age > 30").

Machine learning models encode meaning as **dense floating-point vectors** called *embeddings*. Two texts with similar meaning will have similar vectors, even if they share no keywords. This is why you need a vector database:

- A search for "automobile" should find documents about "car" and "vehicle".
- A search for "I feel unwell" should match "symptoms of illness".
- Traditional keyword search cannot do this.

A vector database stores embeddings and answers **nearest-neighbor queries**: "find the K vectors most similar to this query vector."

---

## 3. Why Brute-Force Is Exact But Slow

The `ExactVectorIndex` computes the cosine similarity between the query vector and **every single stored vector**, then returns the top-k results.

**Why it's exact:** It considers all candidates — it is guaranteed to find the true nearest neighbors.

**Why it's slow:** For N=50,000 vectors and D=384 dimensions, each query requires 50,000 dot-product computations. Query time scales as O(N·D). At 1 million vectors, this becomes 1 billion multiplications per query.

Our implementation uses NumPy's BLAS-accelerated matrix multiplication so it is as fast as a pure O(N) solution can be — but it cannot escape the linear scaling.

---

## 4. Why Approximate Nearest-Neighbor Search?

**Approximate Nearest-Neighbor (ANN)** algorithms accept a small accuracy loss in exchange for dramatically faster search.

Key insight: in practice, the *exact* top-10 neighbors and the *approximate* top-10 neighbors overlap by 90–99%. For most applications, finding 9 of the 10 true nearest neighbors in 1ms is far better than finding all 10 in 50ms.

ANN enables real-time semantic search at scale.

---

## 5. Why HNSW?

HNSW (Hierarchical Navigable Small World) was chosen because:

1. **Best empirical performance** — consistently outperforms competitors (Annoy, IVFPQ) on accuracy–latency tradeoff
2. **Logarithmic search** — empirically approaches O(log N) query time
3. **Online insertion** — new vectors can be added without rebuilding the index
4. **High recall** — with appropriate ef_search, recall@10 routinely exceeds 95%
5. **Influential algorithm** — used inside FAISS, hnswlib, Weaviate, Qdrant, Redis Vector Search

---

## 6. How HNSW Works

HNSW builds a **multilayer navigable small-world graph**:

```
Layer 3:   entry_point ─────────────────────────────> few long-range edges
Layer 2:   ○ ─────── ○ ─────── ○ ─────── ○
Layer 1:   ○ ── ○ ── ○ ── ○ ── ○ ── ○ ── ○
Layer 0:   ○-○-○-○-○-○-○-○-○-○-○-○-○-○-○-○-○ (dense, all nodes)
```

**Level assignment:** When a new node is inserted, its maximum layer `l` is drawn from a geometric distribution: `P(l ≥ k) ≈ (1/M)^k`. Most nodes appear only at layer 0. A few appear at higher layers, creating long-range "highway" connections.

**Insertion:**
1. Greedy descent from the top layer to `l+1`, tracking the current single nearest neighbor.
2. From layer `l` to layer 0: beam search with `ef_construction` candidates → select best M neighbors → connect bidirectionally → prune if neighbor degree exceeds M_max.
3. Update global entry point if `l > max_layer`.

**Search:**
1. Start at the global entry point at the highest layer.
2. Greedy descent (single-NN per step) from top to layer 1.
3. At layer 0: beam search with `ef_search` candidates.
4. Return top-k from layer-0 candidates, excluding deleted nodes.

**Why is it fast?** At high layers, each greedy step covers enormous distance in the data space. By the time the search reaches layer 0, it is already near the true neighbors. Only a small neighborhood needs to be explored at layer 0.

---

## 7. Architecture

```
IT Geeks project/
├── backend/app/
│   ├── main.py                 ← FastAPI entry point
│   ├── api/routes.py           ← All REST endpoints
│   ├── indexes/
│   │   ├── exact_index.py      ← ExactVectorIndex (brute-force)
│   │   ├── hnsw_index.py       ← HNSWIndex (from scratch)
│   │   └── node.py             ← HNSWNode dataclass
│   ├── embeddings/embedder.py  ← all-MiniLM-L6-v2 wrapper
│   ├── benchmark/
│   │   ├── evaluator.py        ← 500-query benchmark runner
│   │   └── metrics.py          ← recall@k, latency, speedup
│   ├── services/index_service.py
│   └── storage/persistence.py  ← .npz + JSON save/load
├── frontend/src/
│   ├── pages/ (5 pages)
│   ├── services/api.ts
│   └── index.css (design system)
├── scripts/
│   ├── generate_dataset.py
│   ├── build_index.py
│   └── run_benchmark.py
├── tests/ (4 test files)
├── data/
│   ├── generated/   ← .npz + JSON (cached, gitignored)
│   └── benchmarks/  ← benchmark_results.json
├── start.py
├── requirements.txt
├── README.md
└── INTERVIEW_NOTES.md
```

---

## 8. Exact Index Implementation

**File:** `backend/app/indexes/exact_index.py`

Key design decisions:
- All vectors stored in a dense NumPy matrix (N×D, float32).
- Vectors are L2-normalized on insertion so dot product = cosine similarity.
- Search uses a single BLAS matrix-vector multiply: `matrix @ query_vector` → O(N) comparisons in one call.
- Deleted nodes are masked with -inf similarity (tombstone).
- `np.argpartition` used for efficient top-k extraction: O(N) instead of O(N log N) full sort.

```python
similarities = self._matrix @ qvec          # single vectorized call
# ... mask deleted rows
top_idx = np.argpartition(similarities, -k)[-k:]
```

---

## 9. HNSW Implementation

**File:** `backend/app/indexes/hnsw_index.py`

Key data structures:
- `_nodes: Dict[str, HNSWNode]` — node storage
- `HNSWNode.neighbors: Dict[int, List[str]]` — per-layer adjacency lists
- `_entry_point: str` — global entry point ID
- `_deleted: Set[str]` — tombstone set

Key algorithms implemented:
- `_random_level()` — geometric distribution level sampling
- `_search_layer()` — heap-based beam search (the core inner loop)
- `_select_neighbors()` — greedy candidate selection
- `_prune_neighbors()` — degree-bound enforcement
- `insert()` — full multilayer insertion with bidirectional edge wiring
- `search()` — greedy descent + layer-0 beam search

---

## 10. Cosine Similarity

```
cosine(A, B) = dot(A, B) / (||A|| × ||B||)
```

When both vectors are L2-normalized (unit vectors): `cosine(A, B) = dot(A, B)`.

Pre-normalizing all stored vectors means every similarity computation reduces to a single dot product. This is the key optimization that makes the brute-force matrix multiply efficient.

Values range from -1 (opposite) to +1 (identical). For semantic embeddings, practical values are 0.5–1.0.

---

## 11. Insert, Search, Delete

### Insert
```
POST /vectors
{"id": "doc_001", "text": "machine learning needs data"}
```
1. Text → embedding (sentence-transformers)
2. `ExactIndex.insert(id, vec, meta)`
3. `HNSWIndex.insert(id, vec, meta)`

### Search
```
POST /search/hnsw
{"text": "neural networks", "top_k": 10, "ef_search": 50}
```
Returns: ranked list of (id, similarity, metadata)

### Delete (Tombstone)
```
DELETE /vectors/doc_001
```
- Sets `node.deleted = True` in both indexes
- Node excluded from all future searches
- HNSW graph edges preserved

---

## 12. Dataset Generation

**Script:** `scripts/generate_dataset.py`

1. Downloads 20 Newsgroups corpus (~18,000 documents, 20 topics)
2. Cleans and chunks each document into 80-word segments with 50% overlap
3. Adds curated topic sentences from 8 domains (tech, science, health, etc.)
4. Augments with controlled text variations to reach 50,000+ unique texts
5. Embeds all texts using `all-MiniLM-L6-v2` (384-dim)
6. Saves to `data/generated/dataset_vectors.npz` and `dataset_metadata.json`

---

## 13. Benchmark Methodology

**Script:** `scripts/run_benchmark.py`  |  **Endpoint:** `POST /benchmark`

1. Sample 500 query vectors from the dataset (fixed seed=42)
2. Warm up both indexes with 5 preliminary queries (excluded from timing)
3. For each query:
   - Time exact search → record `exact_latency_ms` and `exact_ids`
   - Time HNSW search → record `hnsw_latency_ms` and `hnsw_ids`
   - Compute `recall@10 = |exact_ids ∩ hnsw_ids| / 10`
4. Aggregate: mean, median, min, max, std, P95

**Fair benchmarking:**
- Same query vectors for both indexes
- Same cosine similarity metric
- Same active dataset (no ghost vectors)
- Warmup queries excluded
- Index build time reported separately from query latency
- Fixed random seed (reproducible)

---

## 14. Recall@10

```
Recall@10 = |exact_top_10 ∩ hnsw_top_10| / 10
```

- Perfect recall: 10/10 = 1.0
- Missing 1 neighbor: 9/10 = 0.9
- Aggregated over all 500 queries: mean recall@10

**Ground truth is always ExactVectorIndex.** HNSW results are never used as ground truth.

---

## 15. Latency & Speedup

```
Speedup = mean(exact_latency) / mean(HNSW_latency)
```

On a CPU with 50,000 vectors (384-dim):
- Exact search: ~20–100ms (O(N) BLAS matmul)
- HNSW search (ef_search=50): ~1–10ms (graph traversal)
- Typical speedup: 5–50×

Note: on small datasets (<1,000 vectors), brute-force may be faster than HNSW due to graph traversal overhead. Results are reported honestly.

---

## 16. Accuracy vs Latency Tradeoff

| ef_search | Recall@10 | Avg Latency | Speedup |
|-----------|-----------|-------------|---------|
| 10        | ~0.82     | ~0.8ms      | ~25×    |
| 25        | ~0.90     | ~1.5ms      | ~15×    |
| 50        | ~0.95     | ~3ms        | ~10×    |
| 100       | ~0.98     | ~6ms        | ~5×     |
| 200       | ~0.99     | ~12ms       | ~2.5×   |

(Numbers are illustrative; actual values depend on hardware, dataset, and M.)

---

## 17. Limitations

1. **In-memory only:** The full index lives in RAM. For 50,000 × 384-dim float32 vectors: ~73 MB. For 10M vectors: ~14 GB.
2. **Single-threaded HNSW insert:** Python's GIL prevents true parallel insertion. Production implementations use lock-free concurrent insertion.
3. **Tombstone accumulation:** Deleted nodes remain in graph memory until `rebuild()` is called.
4. **No persistence of HNSW graph:** The HNSW graph must be rebuilt on restart (takes a few minutes for 50K vectors).
5. **CPU-bound:** No GPU acceleration. FAISS with GPU support is orders of magnitude faster.
6. **Neighbor selection:** Uses simple greedy neighbor selection; the full HNSW heuristic (`select_neighbors_heuristic`) improves recall for clustered data.

---

## 18. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Generate dataset (one-time, ~20–90 min on CPU)
```bash
python scripts/generate_dataset.py
```

### Step 2: Start API server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Or use the convenience script:
```bash
python start.py
```

### Step 3: Start React frontend
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Step 4 (Optional): Run benchmark
```bash
python scripts/run_benchmark.py
```

### Run tests
```bash
pytest tests/ -v
```

---

### What Is Mocked vs Real

- **Vector Math & Searching**: **0% Mocked**. Both the exact brute-force index and the HNSW graph search are 100% real custom implementations using NumPy.
- **HNSW Graph Construction**: **0% Mocked**. Multilayer entry points, level sampling, greedy routing, beam search, and bidirectional edge updates are implemented from scratch.
- **Ground Truth**: **0% Mocked**. Ground truth top-10 is computed via exact brute-force dot products across the entire active vector space.
- **Dataset Fallback**: If the 50,000 vector background generation script has not finished running yet when the server starts up, the backend automatically initializes an in-memory sample dataset of 1,000 vectors with real text sentences so the API and UI run immediately without throwing 503 errors. Once `generate_dataset.py` finishes, restarting the server loads the full 50,000 vector dataset.
- **HNSW Deletion**: Graph tombstoning (marking nodes as deleted while preserving graph edges to prevent partitioning) is used instead of physical node contraction. As noted in the requirement ("Deletion in a graph index is genuinely awkward — if you skip it, say so and say why"), physical deletion in HNSW requires rewiring neighbors across multiple layers. Tombstoning is the standard approach used in production vector databases (Milvus, Weaviate).

---

## 19. API Documentation

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Index statistics |
| POST | `/vectors` | Insert a vector (text or pre-computed) |
| GET | `/vectors/{id}` | Get vector metadata |
| DELETE | `/vectors/{id}` | Delete (tombstone) |
| POST | `/search/exact` | Exact brute-force search |
| POST | `/search/hnsw` | HNSW approximate search |
| POST | `/benchmark` | Run 500-query benchmark |
| POST | `/benchmark/sweep` | ef_search parameter sweep |
| PUT | `/config/ef_search` | Update ef_search live |

---

## 20. Example Queries

```
"machine learning models need lots of training data"
→ finds: texts about datasets, training, deep learning

"renewable energy sources like solar and wind"
→ finds: texts about climate, power generation, sustainability

"symptoms of fever and fatigue"
→ finds: texts about illness, health, medicine
```

---

## 21. Why I Did Not Use FAISS / Pinecone / Chroma / sklearn.neighbors

The assignment intentionally requires implementing the core indexing logic manually. The purpose is to demonstrate a deep understanding of what happens *inside* a vector database:

- How vectors are stored and normalized
- How cosine similarity is computed efficiently
- How the HNSW multilayer graph is constructed
- How beam search traverses the graph
- Why approximate results diverge from exact results
- How recall and latency are measured and compared

Using an external library would reduce this to a configuration exercise. The custom implementation forces engagement with every design decision: neighbor selection, level sampling, degree pruning, ef_construction vs ef_search, tombstone deletion, and more.

`sklearn` is used *only* to download the 20 Newsgroups raw text corpus. It is not used for any nearest-neighbor computation.

---

## 22. Future Improvements

1. **Concurrent insertion** using a proper read-write lock or lock-free structures
2. **Physical deletion with repair** — reconnect affected neighbors after deletion
3. **HNSW persistence** — serialize graph to binary format (custom or Protocol Buffers)
4. **GPU acceleration** — batch cosine similarity on GPU for exact index
5. **Product Quantization (PQ)** compression to reduce memory footprint 8–32×
6. **Metadata filtering** — filter results by category before/after search
7. **Distributed sharding** — partition the vector space across multiple nodes
8. **IVF (Inverted File Index)** alternative approximate index for comparison
9. **Streaming inserts** — accept real-time text streams and embed on-the-fly
10. **Recall target mode** — automatically tune ef_search to hit a user-specified recall target
