"""
routes.py — All FastAPI route handlers.

Endpoints:
  GET  /health
  GET  /stats
  POST /vectors
  GET  /vectors/{id}
  DELETE /vectors/{id}
  POST /search/exact
  POST /search/hnsw
  POST /benchmark
  POST /benchmark/sweep
  PUT  /config/ef_search
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..models.schemas import (
    BenchmarkRequest,
    DeleteResponse,
    HealthResponse,
    HNSWSearchRequest,
    InsertRequest,
    InsertResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    SweepRequest,
    UpdateEfSearchRequest,
)
from ..services.index_service import IndexService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_service(request: Request) -> IndexService:
    """Dependency injection: retrieve IndexService from app state."""
    return request.app.state.service


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(service: IndexService = Depends(get_service)):
    """Health check endpoint."""
    s = service.stats()
    return HealthResponse(
        status="ok",
        initialized=s["initialized"],
        exact_vectors=s["exact"]["active_vectors"],
        hnsw_vectors=s["hnsw"]["active_vectors"],
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats", tags=["System"])
def stats(service: IndexService = Depends(get_service)):
    """Return comprehensive index statistics."""
    return service.stats()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

@router.post("/vectors", tags=["Vectors"])
def insert_vector(body: InsertRequest, service: IndexService = Depends(get_service)):
    """
    Insert a vector into both indexes.

    Accepts either:
    - text: auto-embeds using the local sentence transformer
    - vector: uses the pre-computed vector directly
    """
    if body.text is None and body.vector is None:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'vector'.")
    if body.text is not None and body.vector is not None:
        raise HTTPException(status_code=400, detail="Provide only one of 'text' or 'vector', not both.")

    try:
        if body.text is not None:
            result = service.insert(body.id, body.text, body.metadata)
        else:
            meta = dict(body.metadata or {})
            result = service.insert_vector(body.id, body.vector, meta)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Insert error")
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

@router.get("/vectors/{id}", tags=["Vectors"])
def get_vector(id: str, service: IndexService = Depends(get_service)):
    """Retrieve metadata for a vector by ID."""
    result = service.get(id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Vector '{id}' not found.")
    return {"id": id, "metadata": result}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/vectors/{id}", tags=["Vectors"])
def delete_vector(id: str, service: IndexService = Depends(get_service)):
    """
    Logically delete a vector from both indexes using tombstoning.

    The vector is marked as deleted and excluded from future searches.
    Graph edges in HNSW are preserved to maintain connectivity.
    """
    result = service.delete(id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Vector '{id}' not found.")
    return result


# ---------------------------------------------------------------------------
# Search — Exact
# ---------------------------------------------------------------------------

@router.post("/search/exact", tags=["Search"])
def search_exact(body: SearchRequest, service: IndexService = Depends(get_service)):
    """
    Exact brute-force cosine similarity search.

    This is the ground-truth search — guaranteed to return the true top-k
    nearest neighbors.  O(N) comparisons per query.
    """
    if body.text is None and body.vector is None:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'vector'.")

    try:
        query = body.text if body.text is not None else body.vector
        result = service.search_exact(query, body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Exact search error")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "results": result["results"],
        "latency_ms": result["latency_ms"],
        "index_type": "exact_brute_force",
        "n_results": len(result["results"]),
        "top_k": body.top_k,
    }


# ---------------------------------------------------------------------------
# Search — HNSW
# ---------------------------------------------------------------------------

@router.post("/search/hnsw", tags=["Search"])
def search_hnsw(body: HNSWSearchRequest, service: IndexService = Depends(get_service)):
    """
    Approximate nearest-neighbor search using the HNSW graph.

    Traverses the multilayer graph — does NOT fall back to brute-force.
    ef_search controls the accuracy-vs-latency tradeoff.
    """
    if body.text is None and body.vector is None:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'vector'.")

    try:
        query = body.text if body.text is not None else body.vector
        result = service.search_hnsw(query, body.top_k, body.ef_search)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("HNSW search error")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "results": result["results"],
        "latency_ms": result["latency_ms"],
        "index_type": "hnsw",
        "n_results": len(result["results"]),
        "top_k": body.top_k,
        "ef_search": body.ef_search or service.hnsw_index.ef_search,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@router.post("/benchmark", tags=["Benchmark"])
def run_benchmark(body: BenchmarkRequest, service: IndexService = Depends(get_service)):
    """
    Run the 500-query benchmark comparing exact vs HNSW.

    Computes recall@k, exact latency, HNSW latency, and speedup.
    Ground truth is always the ExactVectorIndex.
    Results are cached — use force=true to re-run.
    """
    if not service._initialized:
        raise HTTPException(status_code=503, detail="Index not initialized. Run setup first.")
    try:
        results = service.run_benchmark(
            n_queries=body.n_queries,
            k=body.k,
            seed=body.seed,
            force=body.force,
        )
    except Exception as e:
        logger.exception("Benchmark error")
        raise HTTPException(status_code=500, detail=str(e))
    return results


@router.post("/benchmark/sweep", tags=["Benchmark"])
def run_sweep(body: SweepRequest, service: IndexService = Depends(get_service)):
    """
    Run ef_search parameter sweep benchmark.

    Tests multiple ef_search values and measures how recall and latency change.
    Demonstrates the accuracy–latency tradeoff of approximate search.
    """
    if not service._initialized:
        raise HTTPException(status_code=503, detail="Index not initialized. Run setup first.")
    try:
        results = service.run_parameter_sweep(
            ef_search_values=body.ef_search_values,
            n_queries=body.n_queries,
            seed=body.seed,
            force=body.force,
        )
    except Exception as e:
        logger.exception("Sweep error")
        raise HTTPException(status_code=500, detail=str(e))
    return results


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@router.put("/config/ef_search", tags=["Config"])
def update_ef_search(body: UpdateEfSearchRequest, service: IndexService = Depends(get_service)):
    """Dynamically update ef_search for the HNSW index."""
    try:
        service.hnsw_index.update_ef_search(body.ef_search)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ef_search": body.ef_search, "status": "updated"}
