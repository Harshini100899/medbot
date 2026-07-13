"""
backend/memory/redis_memory.py — Short-term conversation memory via Redis

Open-source Redis (Docker) is the authoritative short-term store:
  • rolling per-session message history (TTL)
  • session context / detected language cache
  • retrieval / web-search response cache (Redis TTL)
  • per-session rate limiting

Every helper degrades gracefully to a no-op when Redis is disabled or
unreachable, so the app keeps working (falling back to MongoDB / LangGraph
checkpointer) if the container is not running.
"""
from __future__ import annotations
import json
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_redis_client = None
_redis_unavailable = False   # set once a connection attempt fails, avoids retry storms


async def get_redis():
    """
    Lazily create and cache an async Redis client.

    Returns the client, or ``None`` if Redis is disabled in config or the
    server cannot be reached (short-term memory then silently degrades).
    """
    global _redis_client, _redis_unavailable

    if not settings.REDIS_ENABLED or _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        from redis import asyncio as aioredis

        client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await client.ping()
        _redis_client = client
        logger.info("✅ Redis connected: %s", settings.REDIS_URL)
        return _redis_client
    except Exception as e:
        _redis_unavailable = True
        logger.warning("⚠️  Redis unavailable (%s) — short-term memory disabled.", e)
        return None


async def close_redis() -> None:
    """Close the Redis client on shutdown."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


# ─── Session Context ──────────────────────────────────────────────────────────

async def get_session_context(session_id: str) -> Dict[str, Any]:
    """Retrieve the conversation context for a session."""
    redis = await get_redis()
    if not redis:
        return {}
    try:
        data = await redis.get(f"session:{session_id}:context")
        return json.loads(data) if data else {}
    except Exception as e:
        logger.error(f"Redis get_session_context error: {e}")
        return {}


async def save_session_context(session_id: str, context: Dict[str, Any]) -> None:
    """Save/update session context with TTL."""
    redis = await get_redis()
    if not redis:
        return
    try:
        await redis.setex(
            f"session:{session_id}:context",
            settings.REDIS_TTL,
            json.dumps(context, ensure_ascii=False, default=str),
        )
    except Exception as e:
        logger.error(f"Redis save_session_context error: {e}")


# ─── Message History (short-term, last N turns) ───────────────────────────────

async def push_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session's short-term history list."""
    redis = await get_redis()
    if not redis:
        return
    try:
        key = f"session:{session_id}:messages"
        msg = json.dumps({"role": role, "content": content})
        pipe = redis.pipeline()
        pipe.rpush(key, msg)
        pipe.ltrim(key, -20, -1)          # keep last 20 messages
        pipe.expire(key, settings.REDIS_TTL)
        await pipe.execute()
    except Exception as e:
        logger.error(f"Redis push_message error: {e}")


async def get_messages(session_id: str, last_n: int = 10) -> List[Dict[str, str]]:
    """Return last N messages for a session."""
    redis = await get_redis()
    if not redis:
        return []
    try:
        key = f"session:{session_id}:messages"
        raw = await redis.lrange(key, -last_n, -1)
        return [json.loads(m) for m in raw]
    except Exception as e:
        logger.error(f"Redis get_messages error: {e}")
        return []


# ─── Rate Limiting ────────────────────────────────────────────────────────────

async def check_rate_limit(session_id: str, limit: int = 30, window: int = 60) -> bool:
    """Return True if within rate limit, False if exceeded."""
    redis = await get_redis()
    if not redis:
        return True   # allow if Redis down
    try:
        key = f"ratelimit:{session_id}"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window)
        return count <= limit
    except Exception as e:
        logger.error(f"Redis rate_limit error: {e}")
        return True


# ─── Session Language Cache ───────────────────────────────────────────────────

async def cache_language(session_id: str, language: str) -> None:
    redis = await get_redis()
    if not redis:
        return
    try:
        await redis.setex(f"session:{session_id}:lang", settings.REDIS_TTL, language)
    except Exception:
        pass


async def get_cached_language(session_id: str) -> Optional[str]:
    redis = await get_redis()
    if not redis:
        return None
    try:
        return await redis.get(f"session:{session_id}:lang")
    except Exception:
        return None


# ─── Generic TTL cache (retrieval / web-search results) ───────────────────────

async def cache_get_json(key: str) -> Optional[Any]:
    """Read a cached JSON value (retrieval/web-search cache). None on miss."""
    redis = await get_redis()
    if not redis:
        return None
    try:
        data = await redis.get(f"cache:{key}")
        return json.loads(data) if data else None
    except Exception as e:
        logger.debug("Redis cache_get_json error: %s", e)
        return None


async def cache_set_json(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Store a JSON value with TTL (defaults to REDIS_TTL)."""
    redis = await get_redis()
    if not redis:
        return
    try:
        await redis.setex(
            f"cache:{key}",
            ttl or settings.REDIS_TTL,
            json.dumps(value, ensure_ascii=False, default=str),
        )
    except Exception as e:
        logger.debug("Redis cache_set_json error: %s", e)
