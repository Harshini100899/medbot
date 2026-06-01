"""
backend/api/chat_router.py — Chat REST endpoints
"""
from __future__ import annotations
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from backend.graph.supervisor_graph import process_message
from backend.memory.redis_memory import check_rate_limit, get_messages
from backend.db.mongodb import get_conversation_history, get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID (created if not provided)")
    language_hint: Optional[str] = Field(None, description="Optional language hint: de/en/tr/uk")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I have a headache and fever since yesterday. What should I do?",
                "session_id": None,
            }
        }


class ChatResponse(BaseModel):
    response: str
    session_id: str
    language: str
    intent: str
    agent: str
    is_emergency: bool
    sources: list
    metadata: dict


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message to the medical chatbot.
    Supports DE, EN, TR, UK languages — auto-detected.
    """
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Rate limiting
    if not await check_rate_limit(session_id, limit=30, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before sending more messages.")

    # Process through LangGraph
    result = await process_message(
        user_input=request.message,
        session_id=session_id,
    )

    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        language=result.get("language", "en"),
        intent=result.get("intent", "general"),
        agent=result.get("agent", "unknown"),
        is_emergency=result.get("is_emergency", False),
        sources=result.get("sources", []),
        metadata=result.get("metadata", {}),
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    """Retrieve conversation history for a session."""
    history = await get_conversation_history(session_id, limit=limit)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session metadata."""
    session = await get_session(session_id)
    if not session:
        return {"session_id": session_id, "status": "new"}
    return session


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear short-term memory for a session (Redis)."""
    try:
        import redis.asyncio as aioredis
        from backend.config import settings
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        keys = await r.keys(f"session:{session_id}:*")
        if keys:
            await r.delete(*keys)
        return {"status": "cleared", "session_id": session_id, "keys_removed": len(keys)}
    except Exception as e:
        return {"status": "partial", "error": str(e)}
