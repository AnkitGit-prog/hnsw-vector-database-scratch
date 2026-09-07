"""
embedder.py — Sentence Embedding Wrapper

Uses sentence-transformers (all-MiniLM-L6-v2) to convert text to vectors.
- Model runs entirely locally — no API key or internet needed after first download.
- Embeddings are 384-dimensional float32 vectors.
- The model is loaded once and reused across all calls.
- Raw embeddings are returned as numpy arrays; normalization happens in the indexes.

This module is intentionally separate from the index implementations.
The vector indexes operate purely on vectors; they do not know about text.
"""

from __future__ import annotations

import logging
import time
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Model identifier — lightweight yet high-quality sentence encoder
MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps sentence-transformers to produce 384-dim embeddings from text.

    Usage:
        embedder = Embedder()
        vec = embedder.embed("machine learning is powerful")  # shape (384,)
        vecs = embedder.embed_batch(["text 1", "text 2"])    # shape (2, 384)
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int = 384
        self._load_time: float = 0.0

    def _load(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name} ...")
        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        self._load_time = time.perf_counter() - t0
        logger.info(
            f"Model loaded in {self._load_time:.2f}s | dim={self._dimension}"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Returns numpy array of shape (384,), dtype float32.
        Raw output — NOT yet L2-normalized (indexes handle normalization).
        """
        self._load()
        vec = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(
        self, texts: List[str], batch_size: int = 256, show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of texts in batches.

        Returns numpy array of shape (N, 384), dtype float32.
        """
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "dimension": self._dimension,
            "load_time_seconds": self._load_time,
            "loaded": self._model is not None,
        }
