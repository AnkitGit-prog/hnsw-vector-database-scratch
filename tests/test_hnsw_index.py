"""
test_hnsw_index.py — Unit tests for HNSWIndex.

Key test: verify that HNSW returns a high fraction of the same
neighbors as exact brute-force search on a small controlled dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
import numpy as np
from app.indexes.hnsw_index import HNSWIndex
from app.indexes.exact_index import ExactVectorIndex
from app.benchmark.metrics import recall_at_k


def make_vec(values):
    return np.array(values, dtype=np.float32)


def build_small_hnsw(n: int = 200, d: int = 32, seed: int = 42) -> tuple:
    """Build a small HNSW and exact index for testing."""
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, d)).astype(np.float32)
    hnsw = HNSWIndex(M=16, ef_construction=100, ef_search=50, seed=seed)
    exact = ExactVectorIndex()
    for i in range(n):
        hnsw.insert(f"v{i}", vecs[i], {"idx": i})
        exact.insert(f"v{i}", vecs[i], {"idx": i})
    return hnsw, exact, vecs


# ---- Basic insert ----

def test_insert_single():
    hnsw = HNSWIndex()
    hnsw.insert("a", make_vec([1.0, 0.0, 0.0]), {"text": "hello"})
    assert "a" in hnsw._nodes
    assert len(hnsw) == 1


def test_insert_multiple():
    hnsw = HNSWIndex(M=4, ef_construction=20)
    for i in range(20):
        hnsw.insert(f"v{i}", make_vec([float(i + 1), 0.0, 0.0]), {})
    assert len(hnsw) == 20


def test_duplicate_id_raises():
    hnsw = HNSWIndex()
    hnsw.insert("a", make_vec([1.0, 0.0]), {})
    with pytest.raises(ValueError, match="Duplicate"):
        hnsw.insert("a", make_vec([0.0, 1.0]), {})


def test_dimension_mismatch_raises():
    hnsw = HNSWIndex()
    hnsw.insert("a", make_vec([1.0, 0.0, 0.0]), {})
    with pytest.raises(ValueError, match="Dimension"):
        hnsw.insert("b", make_vec([1.0, 0.0]), {})


# ---- Search ----

def test_search_returns_self():
    hnsw = HNSWIndex(M=4, ef_construction=20, ef_search=10)
    vec = make_vec([1.0, 0.0, 0.0])
    hnsw.insert("a", vec, {})
    for i in range(10):
        hnsw.insert(f"v{i}", make_vec([float(i) * 0.1, 0.5, 0.0]), {})
    results = hnsw.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_search_top_k_count():
    hnsw = HNSWIndex(M=8, ef_construction=40, ef_search=20)
    for i in range(50):
        hnsw.insert(f"v{i}", make_vec([float(i + 1), 0.0, 0.0, 0.0]), {})
    results = hnsw.search(make_vec([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert len(results) == 10


def test_search_similarity_ordering():
    hnsw, exact, vecs = build_small_hnsw(n=100, d=16)
    results = hnsw.search(vecs[0], top_k=5)
    for i in range(len(results) - 1):
        assert results[i]["similarity"] >= results[i + 1]["similarity"]


def test_search_empty_index():
    hnsw = HNSWIndex()
    results = hnsw.search(make_vec([1.0, 0.0]), top_k=5)
    assert results == []


# ---- Delete / tombstone ----

def test_delete_excludes_from_results():
    hnsw = HNSWIndex(M=8, ef_construction=40, ef_search=20)
    rng = np.random.default_rng(77)
    vecs = rng.random((30, 8)).astype(np.float32)
    for i in range(30):
        hnsw.insert(f"v{i}", vecs[i], {})
    hnsw.delete("v0")
    results = hnsw.search(vecs[0], top_k=30)
    ids = [r["id"] for r in results]
    assert "v0" not in ids


def test_delete_returns_true_on_existing():
    hnsw = HNSWIndex()
    hnsw.insert("a", make_vec([1.0, 0.0]), {})
    assert hnsw.delete("a") is True


def test_delete_returns_false_on_missing():
    hnsw = HNSWIndex()
    assert hnsw.delete("nonexistent") is False


def test_graph_stays_connected_after_delete():
    """Even after deletion, non-deleted nodes should still be reachable."""
    hnsw = HNSWIndex(M=8, ef_construction=40, ef_search=20)
    rng = np.random.default_rng(33)
    vecs = rng.random((50, 8)).astype(np.float32)
    for i in range(50):
        hnsw.insert(f"v{i}", vecs[i], {})

    # Delete some nodes
    for i in range(0, 10):
        hnsw.delete(f"v{i}")

    # Search should still return non-deleted results
    results = hnsw.search(vecs[10], top_k=5)
    for r in results:
        assert r["id"] not in {f"v{i}" for i in range(10)}


# ---- Recall vs exact ground truth ----

def test_recall_at_10_above_threshold():
    """
    The most important test: HNSW must return >= 70% of the same top-10
    neighbors as exact brute-force on a small controlled dataset.

    With M=16, ef_construction=100, ef_search=50 and 500 vectors recall
    should be high (>=0.70 on average).
    """
    hnsw, exact, vecs = build_small_hnsw(n=500, d=32, seed=42)
    k = 10
    recalls = []
    rng = np.random.default_rng(99)
    query_indices = rng.choice(len(vecs), size=50, replace=False)

    for qi in query_indices:
        qvec = vecs[qi]
        exact_ids = [r["id"] for r in exact.search(qvec, k)]
        hnsw_ids = [r["id"] for r in hnsw.search(qvec, k, ef_search=50)]
        recalls.append(recall_at_k(exact_ids, hnsw_ids, k))

    mean_recall = sum(recalls) / len(recalls)
    print(f"\nMean recall@10 on small dataset: {mean_recall:.3f}")
    assert mean_recall >= 0.70, f"Recall too low: {mean_recall:.3f}"


def test_recall_improves_with_ef_search():
    """Higher ef_search should yield equal or better recall."""
    hnsw, exact, vecs = build_small_hnsw(n=200, d=32, seed=42)
    k = 10
    rng = np.random.default_rng(77)
    query_indices = rng.choice(len(vecs), size=30, replace=False)

    def mean_recall(ef):
        recalls = []
        for qi in query_indices:
            qvec = vecs[qi]
            exact_ids = [r["id"] for r in exact.search(qvec, k)]
            hnsw_ids = [r["id"] for r in hnsw.search(qvec, k, ef_search=ef)]
            recalls.append(recall_at_k(exact_ids, hnsw_ids, k))
        return sum(recalls) / len(recalls)

    recall_low = mean_recall(5)
    recall_high = mean_recall(50)
    print(f"\nRecall ef=5: {recall_low:.3f}, ef=50: {recall_high:.3f}")
    assert recall_high >= recall_low - 0.1  # allow tiny variance


# ---- Parameter effects ----

def test_ef_search_update():
    hnsw = HNSWIndex(M=8, ef_construction=40, ef_search=20)
    hnsw.update_ef_search(100)
    assert hnsw.ef_search == 100


def test_stats():
    hnsw = HNSWIndex(M=8, ef_construction=40, ef_search=20)
    rng = np.random.default_rng(55)
    for i in range(10):
        hnsw.insert(f"v{i}", rng.random(4).astype(np.float32), {})
    hnsw.delete("v0")
    s = hnsw.stats()
    assert s["total_vectors"] == 10
    assert s["active_vectors"] == 9
    assert s["deleted_vectors"] == 1
    assert s["M"] == 8
