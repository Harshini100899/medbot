"""
backend/db/mongodb.py — MongoDB async persistence layer using Motor
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_motor_client = None
_db = None


async def get_db():
    """Return None immediately to bypass MongoDB usage."""
    return None


# ─── Sessions ─────────────────────────────────────────────────────────────────

async def create_session(session_id: str, meta: Dict = None) -> None:
    db = await get_db()
    if db is None:
        return
    try:
        doc = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "message_count": 0,
            **(meta or {}),
        }
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
    except Exception as e:
        logger.error(f"create_session error: {e}")


async def get_session(session_id: str) -> Optional[Dict]:
    db = await get_db()
    if db is None:
        return None
    try:
        doc = await db.sessions.find_one({"session_id": session_id})
        if doc:
            doc.pop("_id", None)
        return doc
    except Exception as e:
        logger.error(f"get_session error: {e}")
        return None


# ─── Conversations ────────────────────────────────────────────────────────────

async def save_conversation_turn(
    session_id: str,
    user_input: str,
    bot_response: str,
    language: str,
    intent: str,
    agent_used: str,
    sources: List[Dict] = None,
    is_emergency: bool = False,
) -> None:
    db = await get_db()
    if db is None:
        return
    try:
        doc = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc),
            "user_input": user_input,
            "bot_response": bot_response,
            "language": language,
            "intent": intent,
            "agent_used": agent_used,
            "sources": sources or [],
            "is_emergency": is_emergency,
        }
        await db.conversations.insert_one(doc)
        await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$inc": {"message_count": 1},
                "$set": {"updated_at": datetime.now(timezone.utc), "last_language": language},
            },
        )
    except Exception as e:
        logger.error(f"save_conversation_turn error: {e}")


async def get_conversation_history(
    session_id: str, limit: int = 20
) -> List[Dict]:
    db = await get_db()
    if db is None:
        return []
    try:
        cursor = db.conversations.find(
            {"session_id": session_id},
            {"_id": 0},
        ).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return list(reversed(docs))
    except Exception as e:
        logger.error(f"get_conversation_history error: {e}")
        return []


# ─── Doctors ──────────────────────────────────────────────────────────────────

async def search_doctors(
    specialization: Optional[str] = None,
    language: Optional[str] = None,
    city: str = "Oberhausen",
    limit: int = 5,
) -> List[Dict]:
    db = await get_db()
    if db is None:
        return []
    try:
        query: Dict[str, Any] = {"city": city, "available": True}
        if specialization:
            query["specialization"] = {"$regex": specialization, "$options": "i"}
        if language:
            query["languages"] = language

        cursor = db.doctors.find(query, {"_id": 0}).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.error(f"search_doctors error: {e}")
        return []


async def get_hospitals(city: str = "Oberhausen") -> List[Dict]:
    db = await get_db()
    if db is None:
        return []
    try:
        cursor = db.doctors.find(
            {"city": city, "is_hospital": True}, {"_id": 0}
        )
        return await cursor.to_list(length=10)
    except Exception as e:
        logger.error(f"get_hospitals error: {e}")
        return []


# ─── Pharmacies ───────────────────────────────────────────────────────────────

async def get_pharmacies(city: str = "Oberhausen", night_only: bool = False) -> List[Dict]:
    db = await get_db()
    if db is None:
        return []
    try:
        query: Dict[str, Any] = {"city": city}
        if night_only:
            query["night_service"] = True
        cursor = db.pharmacies.find(query, {"_id": 0})
        return await cursor.to_list(length=10)
    except Exception as e:
        logger.error(f"get_pharmacies error: {e}")
        return []


# ─── Health Check ─────────────────────────────────────────────────────────────

async def health_check() -> bool:
    db = await get_db()
    if db is None:
        return False
    try:
        await db.command("ping")
        return True
    except Exception:
        return False
