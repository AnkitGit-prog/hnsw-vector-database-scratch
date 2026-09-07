"""
models.py — Pydantic request/response schemas for the FastAPI routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class InsertRequest(BaseModel):
    id: str = Field(..., description="Unique vector ID")
    text: Optional[str] = Field(None, description="Text to embed (mutually exclusive with vector)")
    vector: Optional[List[float]] = Field(None, description="Pre-computed vector (mutually exclusive with text)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v


class SearchRequest(BaseModel):
    text: Optional[str] = Field(None, description="Natural language query (will be embedded)")
    vector: Optional[List[float]] = Field(None, description="Pre-computed query vector")
    top_k: int = Field(10, ge=1, le=1000, description="Number of results to return")


class HNSWSearchRequest(SearchRequest):
    ef_search: Optional[int] = Field(None, ge=1, le=10000, description="ef_search override")


class BenchmarkRequest(BaseModel):
    n_queries: int = Field(500, ge=10, le=2000, description="Number of benchmark queries")
    k: int = Field(10, ge=1, le=100, description="Recall@k cutoff")
    seed: int = Field(42, description="Random seed for reproducibility")
    force: bool = Field(False, description="Force re-run even if cached results exist")


class SweepRequest(BaseModel):
    ef_search_values: Optional[List[int]] = Field(None, description="List of ef_search values to test")
    n_queries: int = Field(200, ge=10, le=1000)
    seed: int = Field(42)
    force: bool = Field(False)


class UpdateEfSearchRequest(BaseModel):
    ef_search: int = Field(..., ge=1, le=10000)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    id: str
    similarity: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float
    index_type: str
    n_results: int


class InsertResponse(BaseModel):
    id: str
    dimension: int
    status: str


class DeleteResponse(BaseModel):
    id: str
    status: str


class StatsResponse(BaseModel):
    initialized: bool
    exact: Dict[str, Any]
    hnsw: Dict[str, Any]
    embedder: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    exact_vectors: int
    hnsw_vectors: int
