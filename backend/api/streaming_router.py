"""
backend/api/streaming_router.py — Server-Sent Events (SSE) streaming endpoint
Emits named events: token | agent | done | error_event
"""
from __future__ import annotations
import uuid
import json
import logging
from typing import Optional, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.graph.supervisor_graph import process_message
from backend.memory.redis_memory import check_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stream", tags=["Streaming"])


class StreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def _sse(event: str, data: dict) -> str:
    """Format a named SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _event_generator(message: str, session_id: str) -> AsyncIterator[str]:
    """
    Drive the LangGraph pipeline and emit SSE events.

    Strategy: run the full graph (process_message), then simulate streaming
    by chunking the final response — this guarantees all agents/memory/RAG
    run correctly and we still get word-by-word appearance in the UI.
    """
    # Yield a heartbeat so the connection is confirmed immediately
    yield _sse("agent", {"agent": "supervisor"})

    try:
        # Run the full pipeline
        result = await process_message(
            user_input=message,
            session_id=session_id,
        )

        response_text: str = result.get("response", "")
        agent: str        = result.get("agent", "supervisor")
        sources: list     = result.get("sources", [])
        is_emergency: bool = result.get("is_emergency", False)

        # Announce which agent handled it
        yield _sse("agent", {"agent": agent})

        # Stream the response word-by-word for a smooth UX
        words = response_text.split(" ")
        buffer = ""
        for i, word in enumerate(words):
            buffer += word + (" " if i < len(words) - 1 else "")
            # Flush every ~4 words or at newlines
            if (i + 1) % 4 == 0 or "\n" in word:
                yield _sse("token", {"token": buffer})
                buffer = ""

        # Flush remaining
        if buffer:
            yield _sse("token", {"token": buffer})

        # Send done with full metadata
        yield _sse("done", {
            "response":     response_text,
            "session_id":   session_id,
            "agent":        agent,
            "sources":      sources,
            "is_emergency": is_emergency,
            "language":     result.get("language", "en"),
            "intent":       result.get("intent", "general"),
        })

    except Exception as exc:
        logger.exception("SSE stream error: %s", exc)
        yield _sse("error_event", {"message": str(exc)})


@router.get("/chat")
async def stream_chat_get(message: str, session_id: Optional[str] = None):
    """
    Stream chat via Server-Sent Events (GET).

    Connect from JavaScript:
    ```js
    const es = new EventSource(`/api/stream/chat?message=...&session_id=...`);
    es.addEventListener('token', e => console.log(JSON.parse(e.data).token));
    es.addEventListener('done',  e => { ... es.close(); });
    ```
    """
    sid = session_id or str(uuid.uuid4())

    if not await check_rate_limit(sid, limit=30, window=60):
        async def _rate_limit():
            yield _sse("error_event", {"message": "Rate limit exceeded. Please wait."})
        return StreamingResponse(_rate_limit(), media_type="text/event-stream")

    return StreamingResponse(
        _event_generator(message, sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def stream_chat_post(request: StreamRequest):
    """POST version for clients that prefer a request body."""
    sid = request.session_id or str(uuid.uuid4())

    if not await check_rate_limit(sid, limit=30, window=60):
        async def _rate_limit():
            yield _sse("error_event", {"message": "Rate limit exceeded. Please wait."})
        return StreamingResponse(_rate_limit(), media_type="text/event-stream")

    return StreamingResponse(
        _event_generator(request.message, sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
