"""
backend/graph/supervisor_graph.py — Main LangGraph Multi-Agent Graph
P4H Architecture: Supervisor → 6 Specialist Agents → Response Builder
"""
from __future__ import annotations
import logging
import uuid
from typing import AsyncIterator, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from backend.graph.state import MedBotState, INTENTS
from backend.agents.supervisor_agent import run_supervisor, route_to_agent
from backend.agents.emergency_agent import run_emergency_agent
from backend.agents.doctor_search_agent import run_doctor_search_agent
from backend.agents.medical_knowledge_agent import run_medical_knowledge_agent
from backend.agents.policy_rights_agent import run_policy_rights_agent
from backend.agents.location_maps_agent import run_location_maps_agent
from backend.agents.migrant_health_agent import run_migrant_health_agent
from backend.response_builder.builder import build_response
# No Redis/MongoDB imports needed for pure in-memory checkpointer

logger = logging.getLogger(__name__)

# ─── In-memory checkpointer (for LangGraph turn persistence within session) ───
_memory_saver = MemorySaver()


def _build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    workflow = StateGraph(MedBotState)

    # ── Add all nodes ──────────────────────────────────────────────────────
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("emergency_agent", run_emergency_agent)
    workflow.add_node("doctor_search_agent", run_doctor_search_agent)
    workflow.add_node("medical_knowledge_agent", run_medical_knowledge_agent)
    workflow.add_node("policy_rights_agent", run_policy_rights_agent)
    workflow.add_node("location_maps_agent", run_location_maps_agent)
    workflow.add_node("migrant_health_agent", run_migrant_health_agent)
    workflow.add_node("response_builder", build_response)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    # ── Conditional routing from supervisor → agents ───────────────────────
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "emergency_agent": "emergency_agent",
            "doctor_search_agent": "doctor_search_agent",
            "medical_knowledge_agent": "medical_knowledge_agent",
            "policy_rights_agent": "policy_rights_agent",
            "location_maps_agent": "location_maps_agent",
            "migrant_health_agent": "migrant_health_agent",
        },
    )

    # ── All agents → response_builder → END ───────────────────────────────
    for agent_node in [
        "emergency_agent",
        "doctor_search_agent",
        "medical_knowledge_agent",
        "policy_rights_agent",
        "location_maps_agent",
        "migrant_health_agent",
    ]:
        workflow.add_edge(agent_node, "response_builder")

    workflow.add_edge("response_builder", END)

    return workflow.compile(checkpointer=_memory_saver)


# ─── Compiled graph singleton ─────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
        logger.info("✅ LangGraph compiled successfully")
    return _graph


# ─── Public API ───────────────────────────────────────────────────────────────

async def process_message(
    user_input: str,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """
    Process a user message through the full agent pipeline.

    Returns
    -------
    {
        "response": str,
        "session_id": str,
        "metadata": dict,
        "sources": list,
        "is_emergency": bool,
    }
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    graph = get_graph()

    # ── Initial state (In-memory LangGraph checkpointer will auto-append message) ──
    initial_state: MedBotState = {
        "messages": [HumanMessage(content=user_input)],
        "session_id": session_id,
        "user_input": user_input,
        "user_language": "en",
        "language_confidence": 0.0,
        "detected_intent": "general",
        "intent_confidence": 0.0,
        "active_agent": "",
        "agent_raw_output": "",
        "sources": [],
        "retrieved_docs": [],
        "is_emergency": False,
        "needs_disclaimer": True,
        "needs_maps": False,
        "normalised_terms": [],
        "final_response": "",
        "response_metadata": {},
        "extra_context": {},
    }

    # ── Run the graph ─────────────────────────────────────────────────────
    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"Graph execution error: {e}", exc_info=True)
        return {
            "response": "I apologise, a system error occurred. Please try again.",
            "session_id": session_id,
            "metadata": {"error": str(e)},
            "sources": [],
            "is_emergency": False,
        }

    response_text = final_state.get("final_response", "")
    metadata = final_state.get("response_metadata", {})
    sources = final_state.get("sources", [])
    is_emergency = final_state.get("is_emergency", False)
    lang = final_state.get("user_language", "en")
    intent = final_state.get("detected_intent", "general")
    agent = final_state.get("active_agent", "unknown")

    return {
        "response": response_text,
        "session_id": session_id,
        "language": lang,
        "intent": intent,
        "agent": agent,
        "metadata": metadata,
        "sources": sources,
        "is_emergency": is_emergency,
    }


async def stream_message(
    user_input: str,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """
    Stream the response token by token (uses LangGraph astream_events).
    Yields SSE-compatible string chunks.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    graph = get_graph()

    initial_state: MedBotState = {
        "messages": [HumanMessage(content=user_input)],
        "session_id": session_id,
        "user_input": user_input,
        "user_language": "en",
        "language_confidence": 0.0,
        "detected_intent": "general",
        "intent_confidence": 0.0,
        "active_agent": "",
        "agent_raw_output": "",
        "sources": [],
        "retrieved_docs": [],
        "is_emergency": False,
        "needs_disclaimer": True,
        "needs_maps": False,
        "normalised_terms": [],
        "final_response": "",
        "response_metadata": {},
        "extra_context": {},
    }

    config = {"configurable": {"thread_id": session_id}}

    full_response = ""
    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")
            # Stream LLM token output from agent nodes
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_response += token
                    yield token
            # Final state — get the complete formatted response
            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final = output.get("final_response", "")
                    if final and final != full_response:
                        # Yield the remainder (disclaimer, sources, etc.)
                        remainder = final[len(full_response):]
                        if remainder:
                            yield remainder
                        full_response = final
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"\n\n[Error: {e}]"


def clear_graph_memory(session_id: str) -> bool:
    """Delete checkpoints and writes associated with the session_id from LangGraph memory."""
    try:
        _memory_saver.delete_thread(session_id)
        logger.info(f"Cleared LangGraph memory for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error clearing LangGraph memory for session {session_id}: {e}")
        return False


async def list_graph_sessions() -> list:
    """List all active sessions stored in LangGraph MemorySaver."""
    sessions = []
    graph = get_graph()
    thread_ids = list(_memory_saver.storage.keys())
    for tid in thread_ids:
        try:
            state = await graph.aget_state({"configurable": {"thread_id": tid}})
            if not state or not state.values:
                continue
            messages = state.values.get("messages", [])
            if not messages:
                continue
            # Get first message content for title
            first_msg = messages[0].content
            title = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
            last_msg = messages[-1].content
            updated_at = state.created_at or ""
            sessions.append({
                "session_id": tid,
                "title": title,
                "last_message": last_msg,
                "updated_at": updated_at,
                "message_count": len(messages),
            })
        except Exception as e:
            logger.error(f"Error reading session {tid} details: {e}")
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions
