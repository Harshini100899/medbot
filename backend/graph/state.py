"""
backend/graph/state.py — LangGraph shared state for all agents
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─── Intent types the Supervisor can classify ─────────────────────────────────
INTENTS = {
    "emergency":        "emergency_agent",
    "doctor_search":    "doctor_search_agent",
    "medical_knowledge":"medical_knowledge_agent",
    "policy_rights":    "policy_rights_agent",
    "location_maps":    "medical_knowledge_agent",   # disabled maps_agent, route to medical_knowledge
    "migrant_health":   "migrant_health_agent",
    "general":          "medical_knowledge_agent",   # default fallback
}


class MedBotState(TypedDict, total=False):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]  # full message history
    session_id: str
    user_input: str                  # raw user message (current turn)

    # ── Language ──────────────────────────────────────────────────────────────
    user_language: str               # detected ISO-639-1 code: de/en/tr/uk
    language_confidence: float

    # ── Intent / Routing ──────────────────────────────────────────────────────
    detected_intent: str             # one of INTENTS keys
    intent_confidence: float
    active_agent: str                # agent currently handling the request

    # ── Agent Output ──────────────────────────────────────────────────────────
    agent_raw_output: str            # answer from specialist agent
    sources: List[Dict[str, Any]]    # cited sources [{title, url, snippet}]
    retrieved_docs: List[str]        # RAG documents used

    # ── Flags ─────────────────────────────────────────────────────────────────
    is_emergency: bool               # triggers emergency banner
    needs_disclaimer: bool           # medical disclaimer needed
    needs_maps: bool                 # should attach maps/directions

    # ── Ontology Normalisation ────────────────────────────────────────────────
    normalised_terms: List[str]      # SNOMED / ICD-10 terms found

    # ── Final Response ────────────────────────────────────────────────────────
    final_response: str              # assembled, translated, formatted response
    response_metadata: Dict[str, Any]# timing, agent used, confidence, etc.

    # ── Context (pass-through between nodes) ─────────────────────────────────
    extra_context: Dict[str, Any]    # arbitrary node-to-node data
