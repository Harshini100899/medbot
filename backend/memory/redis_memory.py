"""
backend/memory/redis_memory.py — Short-term conversation memory via Redis
"""
from __future__ import annotations
import json
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_redis_client = None


async def get_redis():
    """Return None immediately to bypass Redis usage."""
    return None


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
