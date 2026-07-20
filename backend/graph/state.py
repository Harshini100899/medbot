"""
backend/graph/state.py — LangGraph shared state for all agents
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _reset_or_append(
    existing: List[Dict[str, Any]], new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reducer for ``agent_outputs``: concatenates within a turn, but treats an
    empty write as an explicit reset (LangGraph's BinaryOperatorAggregate folds
    every update onto the persisted checkpoint value, so a plain ``operator.add``
    write of ``[]`` would be a no-op and stale entries would leak into the next
    conversation turn on the same checkpointed thread)."""
    if not new:
        return []
    return existing + new


# ─── Hierarchical routing (matches the 3-level architecture) ──────────────────
#
#   Level 1 — Supervisor (gateway router): cheap binary/tertiary classification
#             into one of TOP_ROUTES.
#   Level 2 — Medical Specialist  (clinical) OR  General Purpose (orchestrator).
#   Level 3 — The General Purpose Agent dispatches to one of four sub-agents.
#
TOP_ROUTES = {
    "emergency": "emergency_agent",          # fast-path, bypasses specialists
    "medical":   "medical_specialist",       # clinical knowledge agent
    "general":   "general_purpose",          # orchestrator over the sub-agents
}

# Level-3 sub-intents handled by the General Purpose orchestrator.
GENERAL_SUBINTENTS = {
    "doctor_search":  "doctor_search_agent",
    "policy_rights":  "policy_rights_agent",
    "location_maps":  "location_maps_agent",
    "migrant_health": "migrant_health_agent",
}

# Legacy flat map kept for any callers that still classify a single intent.
INTENTS = {
    "emergency":         "emergency_agent",
    "doctor_search":     "doctor_search_agent",
    "medical_knowledge": "medical_specialist",
    "policy_rights":     "policy_rights_agent",
    "location_maps":     "location_maps_agent",
    "migrant_health":    "migrant_health_agent",
    "general":           "medical_specialist",   # default fallback
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
    top_level_route: str             # Level 1: emergency | medical | general
    detected_intent: str             # Level 3 primary sub-intent (GENERAL_SUBINTENTS keys)
    active_intents: List[str]        # Level 3 sub-intents selected this turn (fan-out capable)
    intent_confidence: float
    active_agent: str                # agent(s) that handled the request, "+"-joined if multiple

    # ── Agent Output ────────────────────────────────────────────────────────────
    # Single-writer path: used by emergency_agent / medical_knowledge_agent, which
    # never run in parallel with another agent in the same turn.
    agent_raw_output: str            # answer from specialist agent
    sources: List[Dict[str, Any]]    # cited sources [{title, url, snippet}]
    retrieved_docs: List[str]        # RAG documents used

    # Fan-out path: each General Purpose sub-agent appends one entry here instead
    # of writing the flat fields above, since 2+ of them may run in the same graph
    # super-step (LangGraph forbids concurrent writes to un-annotated keys). Reset
    # to [] by response_builder every turn so it doesn't leak across conversation
    # turns on the checkpointed thread.
    agent_outputs: Annotated[List[Dict[str, Any]], _reset_or_append]

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
