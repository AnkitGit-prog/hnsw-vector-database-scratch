"""
main.py — FastAPI application entry point.

Starts the vector database API server.
Loads both indexes from disk on startup.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.api.routes import router
from app.services.index_service import IndexService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load indexes on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("  Vector Database API — Starting Up")
    logger.info("=" * 60)

    service = IndexService(M=16, ef_construction=200, ef_search=50)
    app.state.service = service

    initialized = service.initialize()
    if initialized:
        s = service.stats()
        logger.info(
            f"Loaded {s['exact']['active_vectors']:,} vectors | "
            f"dim={s['exact']['dimension']} | "
            f"HNSW max_layer={s['hnsw']['max_layer']}"
        )
    else:
        logger.warning(
            "No dataset found. API is running but indexes are empty. "
            "Run: python scripts/generate_dataset.py && python scripts/build_index.py"
        )

    yield

    logger.info("Vector Database API — Shutting down.")


app = FastAPI(
    title="Vector Database From Scratch",
    description=(
        "A custom vector search engine implementing ExactVectorIndex (brute-force) "
        "and HNSWIndex (hierarchical navigable small world) from scratch using NumPy. "
        "No FAISS, Pinecone, Chroma, or sklearn.neighbors used."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["System"])
def root():
    return {
        "name": "Vector Database From Scratch",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
    }
