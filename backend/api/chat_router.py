"""
backend/api/chat_router.py — Chat REST endpoints
"""
from __future__ import annotations
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.graph.supervisor_graph import process_message, get_graph, clear_graph_memory, list_graph_sessions

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
        agent="supervisor",  # Force UI to display Supervisor Agent
        is_emergency=result.get("is_emergency", False),
        sources=result.get("sources", []),
        metadata=result.get("metadata", {}),
    )


@router.get("/sessions")
async def get_all_sessions():
    """Retrieve all active chat sessions stored in LangGraph Memory."""
    try:
        sessions = await list_graph_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return {"sessions": []}


@router.post("/sessions")
async def create_new_session():
    """Explicitly generate a new session ID."""
    new_id = str(uuid.uuid4())
    return {"session_id": new_id, "status": "created"}


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a session directly from LangGraph."""
    try:
        graph = get_graph()
        state = await graph.aget_state({"configurable": {"thread_id": session_id}})
        messages = state.values.get("messages", []) if state and state.values else []
        
        history = []
        for m in messages:
            role = "user" if m.type == "human" else "assistant"
            history.append({
                "role": role,
                "content": m.content,
                "is_emergency": getattr(m, "additional_kwargs", {}).get("is_emergency", False),
                "agent_used": "supervisor"
            })
        return {"session_id": session_id, "history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error getting history for {session_id}: {e}")
        return {"session_id": session_id, "history": [], "count": 0, "error": str(e)}


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session metadata from LangGraph checkpointer."""
    try:
        graph = get_graph()
        state = await graph.aget_state({"configurable": {"thread_id": session_id}})
        if state and state.values:
            return {
                "session_id": session_id,
                "status": "active",
                "created_at": state.created_at,
                "message_count": len(state.values.get("messages", []))
            }
        return {"session_id": session_id, "status": "new"}
    except Exception:
        return {"session_id": session_id, "status": "new"}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear memory checkpoints for the chat session entirely."""
    ok = clear_graph_memory(session_id)
    if ok:
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=500, detail="Failed to delete session memory")


@router.post("/session/{session_id}/clear")
async def clear_chat_history(session_id: str):
    """Clear memory checkpoints for the chat session, starting it fresh."""
    ok = clear_graph_memory(session_id)
    if ok:
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=500, detail="Failed to clear session memory")
