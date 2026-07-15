"""
backend/graph/supervisor_graph.py — Main LangGraph Multi-Agent Graph

Hierarchical (3-level) architecture:

    Supervisor (Level 1 gateway router)
        ├─ emergency  → Emergency Agent (fast-path)
        ├─ medical    → Medical Specialist Agent (clinical knowledge)
        └─ general    → General Purpose Agent (orchestrator)
                            ├─ doctor_search  → Doctor Search Agent
                            ├─ policy_rights  → Policy & Rights Agent
                            ├─ location_maps  → Maps Agent
                            └─ migrant_health → Migrant Health Agent
        └─ (all) → Response Builder → END

Memory:
    • Redis    — short-term rolling history + language/response cache (authoritative)
    • MongoDB  — durable conversation history + sessions (authoritative)
    • LangGraph MemorySaver — in-process per-turn checkpointer (fallback)

Observability:
    • Langfuse — env-gated LangChain callback traces the whole run.
"""
from __future__ import annotations
import logging
import uuid
from typing import AsyncIterator, Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend.graph.state import MedBotState, TOP_ROUTES, GENERAL_SUBINTENTS
from backend.agents.supervisor_agent import run_supervisor, route_to_agent
from backend.agents.general_purpose_agent import run_general_purpose_agent, route_to_subagent
from backend.agents.emergency_agent import run_emergency_agent
from backend.agents.medical_knowledge_agent import run_medical_knowledge_agent
from backend.subagents.doctor_search_agent import run_doctor_search_agent
from backend.subagents.policy_rights_agent import run_policy_rights_agent
from backend.subagents.location_maps_agent import run_location_maps_agent
from backend.subagents.migrant_health_agent import run_migrant_health_agent
from backend.response_builder.builder import build_response

from backend.memory.redis_memory import check_rate_limit, push_message
from backend.db.mongodb import (
    create_session,
    save_conversation_turn,
    get_conversation_history,
)
from backend.observability.langfuse_tracer import get_langfuse_handler, run_metadata

logger = logging.getLogger(__name__)

# ─── In-memory checkpointer (per-turn persistence, fallback within a process) ──
_memory_saver = MemorySaver()


def _build_graph() -> StateGraph:
    """Construct and compile the hierarchical LangGraph StateGraph."""
    workflow = StateGraph(MedBotState)

    # ── Nodes ───────────────────────────────────────────────────────────────
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("general_purpose", run_general_purpose_agent)
    workflow.add_node("emergency_agent", run_emergency_agent)
    workflow.add_node("medical_specialist", run_medical_knowledge_agent)
    workflow.add_node("doctor_search_agent", run_doctor_search_agent)
    workflow.add_node("policy_rights_agent", run_policy_rights_agent)
    workflow.add_node("location_maps_agent", run_location_maps_agent)
    workflow.add_node("migrant_health_agent", run_migrant_health_agent)
    workflow.add_node("response_builder", build_response)

    workflow.set_entry_point("supervisor")

    # ── Level 1: supervisor → {emergency | medical | general} ────────────────
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "emergency_agent": "emergency_agent",
            "medical_specialist": "medical_specialist",
            "general_purpose": "general_purpose",
        },
    )

    # ── Level 3: general_purpose orchestrator → one of four sub-agents ───────
    workflow.add_conditional_edges(
        "general_purpose",
        route_to_subagent,
        {
            "doctor_search_agent": "doctor_search_agent",
            "policy_rights_agent": "policy_rights_agent",
            "location_maps_agent": "location_maps_agent",
            "migrant_health_agent": "migrant_health_agent",
        },
    )

    # ── Every answering agent → response_builder → END ──────────────────────
    for agent_node in [
        "emergency_agent",
        "medical_specialist",
        "doctor_search_agent",
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
        logger.info("✅ LangGraph compiled successfully (hierarchical 3-level)")
    return _graph


# ─── Memory helpers ───────────────────────────────────────────────────────────

async def _has_checkpoint(session_id: str) -> bool:
    """True if the LangGraph checkpointer already holds this session's messages."""
    try:
        state = await get_graph().aget_state({"configurable": {"thread_id": session_id}})
        return bool(state and state.values and state.values.get("messages"))
    except Exception:
        return False


async def _hydrate_messages(session_id: str) -> List[BaseMessage]:
    """
    Rebuild prior turns from MongoDB (durable) so context survives restarts.
    Returns an ordered list of Human/AI messages (may be empty).
    """
    msgs: List[BaseMessage] = []
    try:
        history = await get_conversation_history(session_id, limit=10)
        for turn in history:
            if turn.get("user_input"):
                msgs.append(HumanMessage(content=turn["user_input"]))
            if turn.get("bot_response"):
                msgs.append(AIMessage(content=turn["bot_response"]))
    except Exception as e:
        logger.debug("History hydration skipped: %s", e)
    return msgs


async def _initial_state(session_id: str, user_input: str) -> MedBotState:
    """Build the initial graph state, hydrating history on a cold checkpointer."""
    if await _has_checkpoint(session_id):
        # add_messages will append this turn to the existing checkpoint history.
        messages: List[BaseMessage] = [HumanMessage(content=user_input)]
    else:
        # Cold start (e.g. after restart) — seed durable history from Mongo.
        messages = await _hydrate_messages(session_id)
        messages.append(HumanMessage(content=user_input))

    return {
        "messages": messages,
        "session_id": session_id,
        "user_input": user_input,
        "user_language": "en",
        "language_confidence": 0.0,
        "top_level_route": "medical",
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


def _run_config(session_id: str) -> Dict[str, Any]:
    """Graph run config: thread id + optional Langfuse callback."""
    config: Dict[str, Any] = {"configurable": {"thread_id": session_id}}
    handler = get_langfuse_handler(
        session_id=session_id,
        metadata={"application": "p4h-medbot"},
    )
    if handler is not None:
        config["callbacks"] = [handler]
        config["metadata"] = run_metadata(
            session_id=session_id,
            metadata={"application": "p4h-medbot"},
        )
    return config


async def _persist_turn(session_id: str, user_input: str, result: Dict[str, Any]) -> None:
    """Write the completed turn to Redis (short-term) and MongoDB (durable)."""
    response = result.get("response", "")
    lang = result.get("language", "en")
    intent = result.get("intent", "general")
    agent = result.get("agent", "unknown")
    sources = result.get("sources", [])
    is_emergency = result.get("is_emergency", False)

    try:
        await push_message(session_id, "user", user_input)
        await push_message(session_id, "assistant", response)
    except Exception as e:
        logger.debug("Redis push skipped: %s", e)

    try:
        await create_session(session_id, {"last_language": lang})
        await save_conversation_turn(
            session_id=session_id,
            user_input=user_input,
            bot_response=response,
            language=lang,
            intent=intent,
            agent_used=agent,
            sources=sources,
            is_emergency=is_emergency,
        )
    except Exception as e:
        logger.debug("Mongo persist skipped: %s", e)


# ─── Public API ───────────────────────────────────────────────────────────────

async def process_message(
    user_input: str,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """Process a user message through the full hierarchical agent pipeline."""
    if not session_id:
        session_id = str(uuid.uuid4())

    # ── Rate limiting (Redis-backed; allows through if Redis is down) ───────
    if not await check_rate_limit(session_id):
        return {
            "response": "You are sending messages too quickly. Please wait a moment and try again.",
            "session_id": session_id,
            "language": "en",
            "intent": "rate_limited",
            "agent": "supervisor",
            "metadata": {"rate_limited": True},
            "sources": [],
            "is_emergency": False,
        }

    graph = get_graph()
    initial_state = await _initial_state(session_id, user_input)
    config = _run_config(session_id)

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

    result = {
        "response": final_state.get("final_response", ""),
        "session_id": session_id,
        "language": final_state.get("user_language", "en"),
        "intent": final_state.get("detected_intent", "general"),
        "agent": final_state.get("active_agent", "unknown"),
        "metadata": final_state.get("response_metadata", {}),
        "sources": final_state.get("sources", []),
        "is_emergency": final_state.get("is_emergency", False),
    }

    await _persist_turn(session_id, user_input, result)
    return result


async def stream_message(
    user_input: str,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """Stream the response token by token (LangGraph astream_events)."""
    if not session_id:
        session_id = str(uuid.uuid4())

    if not await check_rate_limit(session_id):
        yield "You are sending messages too quickly. Please wait a moment and try again."
        return

    graph = get_graph()
    initial_state = await _initial_state(session_id, user_input)
    config = _run_config(session_id)

    full_response = ""
    final_text = ""
    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_response += token
                    yield token
            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final = output.get("final_response", "")
                    if final:
                        final_text = final
                        if final != full_response:
                            remainder = final[len(full_response):]
                            if remainder:
                                yield remainder
                            full_response = final
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"\n\n[Error: {e}]"
        return

    # Persist the streamed turn (best-effort).
    await _persist_turn(
        session_id,
        user_input,
        {"response": final_text or full_response, "agent": "supervisor"},
    )


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
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions
