"""
backend/memory/chroma_memory.py — Long-term vector memory via ChromaDB + RAG
"""
from __future__ import annotations
import logging
import os
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_chroma_client = None
_collections: Dict[str, Any] = {}
_embedding_fn = None


def _get_embedding_function():
    """Return None to bypass embedding loading."""
    return None


def get_chroma_client():
    """Return None to bypass ChromaDB connection."""
    return None


def get_collection(name: str):
    """Get or create a ChromaDB collection."""
    global _collections
    if name in _collections:
        return _collections[name]
    client = get_chroma_client()
    if not client:
        return None
    try:
        emb_fn = _get_embedding_function()
        col = client.get_or_create_collection(
            name=name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        _collections[name] = col
        return col
    except Exception as e:
        logger.error(f"Collection '{name}' error: {e}")
        return None


# ─── Collections ──────────────────────────────────────────────────────────────
MEDICAL_COLLECTION = "medical_knowledge"
POLICY_COLLECTION = "policy_rights"
SESSION_COLLECTION = "session_summaries"


# ─── Add Documents ────────────────────────────────────────────────────────────

def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
) -> bool:
    """Add documents to a ChromaDB collection."""
    col = get_collection(collection_name)
    if not col:
        return False
    try:
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{}] * len(documents)
        col.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return True
    except Exception as e:
        logger.error(f"add_documents error: {e}")
        return False


# ─── Query / RAG ──────────────────────────────────────────────────────────────

def query_documents(
    collection_name: str,
    query: str,
    n_results: int = None,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Semantic search in a collection. Returns [{text, metadata, score}]."""
    col = get_collection(collection_name)
    if not col:
        return []
    n = n_results or settings.RAG_TOP_K
    try:
        kwargs = {"query_texts": [query], "n_results": min(n, col.count() or 1)}
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist   # cosine similarity (0–1)
            if score >= settings.RAG_SCORE_THRESHOLD:
                out.append({"text": doc, "metadata": meta, "score": round(score, 4)})
        return out
    except Exception as e:
        logger.error(f"query_documents error: {e}")
        return []


# ─── Store Session Summary ────────────────────────────────────────────────────

def save_session_summary(session_id: str, summary: str, language: str) -> None:
    """Persist a session summary for long-term recall."""
    add_documents(
        SESSION_COLLECTION,
        [summary],
        metadatas=[{"session_id": session_id, "language": language}],
        ids=[f"sess_{session_id}"],
    )


def search_past_sessions(query: str, n: int = 3) -> List[Dict]:
    return query_documents(SESSION_COLLECTION, query, n_results=n)


# ─── Collection stats ─────────────────────────────────────────────────────────

def collection_count(name: str) -> int:
    col = get_collection(name)
    return col.count() if col else 0
