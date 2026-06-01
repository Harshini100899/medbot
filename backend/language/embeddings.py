"""
backend/language/embeddings.py — Multilingual sentence embeddings
"""
from __future__ import annotations
import logging
from typing import List, Optional
from functools import lru_cache

from backend.config import settings

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model():
    """Lazy-load the sentence-transformer model (cached)."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info(f"✅ Embedding model loaded: {settings.EMBEDDING_MODEL}")
        return _model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def embed_text(text: str) -> Optional[List[float]]:
    """Embed a single string. Returns list of floats or None on failure."""
    model = get_embedding_model()
    if not model:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.error(f"embed_text error: {e}")
        return None


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch embed multiple strings."""
    model = get_embedding_model()
    if not model:
        return None
    try:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.error(f"embed_texts error: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
