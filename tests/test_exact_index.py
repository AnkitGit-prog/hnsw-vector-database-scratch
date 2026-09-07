"""
test_exact_index.py — Unit tests for ExactVectorIndex.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import math
import pytest
import numpy as np
from app.indexes.exact_index import ExactVectorIndex


def make_vec(values):
    return np.array(values, dtype=np.float32)


# ---- Basic insert and search ----

def test_insert_and_search_returns_self():
    idx = ExactVectorIndex()
    vec = make_vec([1.0, 0.0, 0.0])
    idx.insert("a", vec, {"text": "hello"})
    results = idx.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"
    assert abs(results[0]["similarity"] - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    idx = ExactVectorIndex()
    idx.insert("a", make_vec([1.0, 0.0]), {})
    idx.insert("b", make_vec([0.0, 1.0]), {})
    results = idx.search(make_vec([1.0, 0.0]), top_k=2)
    # a should be first (sim=1.0), b second (sim=0.0)
    assert results[0]["id"] == "a"
    assert abs(results[0]["similarity"] - 1.0) < 1e-5
    assert abs(results[1]["similarity"]) < 1e-5


def test_top_k_ordering():
    idx = ExactVectorIndex()
    idx.insert("close", make_vec([0.99, 0.01]), {})
    idx.insert("mid",   make_vec([0.7, 0.3]), {})
    idx.insert("far",   make_vec([0.1, 0.9]), {})
    results = idx.search(make_vec([1.0, 0.0]), top_k=3)
    assert results[0]["id"] == "close"
    assert results[1]["id"] == "mid"
    assert results[2]["id"] == "far"
    # Similarities should be decreasing
    for i in range(len(results) - 1):
        assert results[i]["similarity"] >= results[i + 1]["similarity"]


def test_top_k_limited():
    idx = ExactVectorIndex()
    for i in range(10):
        idx.insert(f"v{i}", make_vec([float(i + 1), 0.0, 0.0]), {})
    results = idx.search(make_vec([1.0, 0.0, 0.0]), top_k=3)
    assert len(results) == 3


def test_delete_excludes_from_results():
    idx = ExactVectorIndex()
    idx.insert("a", make_vec([1.0, 0.0]), {})
    idx.insert("b", make_vec([0.9, 0.1]), {})
    idx.delete("a")
    results = idx.search(make_vec([1.0, 0.0]), top_k=5)
    ids = [r["id"] for r in results]
    assert "a" not in ids
    assert "b" in ids


def test_delete_returns_true_on_existing():
    idx = ExactVectorIndex()
    idx.insert("x", make_vec([1.0, 0.0]), {})
    assert idx.delete("x") is True


def test_delete_returns_false_on_missing():
    idx = ExactVectorIndex()
    assert idx.delete("nonexistent") is False


# ---- Validation ----

def test_duplicate_id_raises():
    idx = ExactVectorIndex()
    idx.insert("a", make_vec([1.0, 0.0]), {})
    with pytest.raises(ValueError, match="Duplicate"):
        idx.insert("a", make_vec([0.0, 1.0]), {})


def test_dimension_mismatch_raises():
    idx = ExactVectorIndex()
    idx.insert("a", make_vec([1.0, 0.0, 0.0]), {})
    with pytest.raises(ValueError, match="Dimension"):
        idx.insert("b", make_vec([1.0, 0.0]), {})


def test_empty_vector_raises():
    idx = ExactVectorIndex()
    with pytest.raises(ValueError, match="empty"):
        idx.insert("a", make_vec([]), {})


def test_nan_vector_raises():
    idx = ExactVectorIndex()
    with pytest.raises(ValueError, match="NaN"):
        idx.insert("a", make_vec([float("nan"), 1.0]), {})


def test_inf_vector_raises():
    idx = ExactVectorIndex()
    with pytest.raises(ValueError, match="NaN"):
        idx.insert("a", make_vec([float("inf"), 1.0]), {})


def test_zero_vector_raises():
    idx = ExactVectorIndex()
    with pytest.raises(ValueError, match="zero"):
        idx.insert("a", make_vec([0.0, 0.0]), {})


# ---- Stats ----

def test_stats():
    idx = ExactVectorIndex()
    for i in range(5):
        idx.insert(f"v{i}", make_vec([float(i + 1), 0.0]), {})
    idx.delete("v0")
    s = idx.stats()
    assert s["total_vectors"] == 5
    assert s["active_vectors"] == 4
    assert s["deleted_vectors"] == 1
    assert s["dimension"] == 2


# ---- Empty index ----

def test_search_empty_index():
    idx = ExactVectorIndex()
    results = idx.search(make_vec([1.0, 0.0]), top_k=5)
    assert results == []


# ---- Many vectors ----

def test_large_index_correctness():
    """Verify that brute-force always returns exact nearest neighbor."""
    rng = np.random.default_rng(42)
    idx = ExactVectorIndex()
    n, d = 1000, 64
    vecs = rng.random((n, d)).astype(np.float32)

    for i in range(n):
        idx.insert(f"v{i}", vecs[i], {"idx": i})

    # Query with first vector — should be its own nearest neighbor
    results = idx.search(vecs[0], top_k=1)
    assert results[0]["id"] == "v0"
