"""
backend/agents/general_purpose_agent.py — General Purpose Agent (Orchestrator)

Level-2 node under the Supervisor. It does NOT answer the user directly;
instead it classifies the municipal / logistical request into one of four
sub-intents and hands off (via a conditional edge) to the matching sub-agent:

    doctor_search  → Doctor Search Agent
    policy_rights  → Policy & Rights Agent
    location_maps  → Maps Agent
    migrant_health → Migrant Health Agent

Uses the shared keyword rules for the obvious cases (cheap, no LLM) and a small
LLM classification for the ambiguous ones.
"""
from __future__ import annotations
import json
import logging
import re

from langchain_core.messages import SystemMessage

from backend.graph.state import MedBotState, GENERAL_SUBINTENTS
from backend.agents.supervisor_agent import rule_based_intent
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

DEFAULT_SUBINTENT = "doctor_search"

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for the general-purpose branch of a
medical assistant in Oberhausen, Germany. Route the user's message to EXACTLY ONE sub-agent:

- doctor_search  : Find a doctor, specialist, clinic, hospital or pharmacy; book an appointment.
- policy_rights  : Health insurance (GKV/PKV), asylum health rights, prescriptions, coverage,
                   administrative/legal healthcare questions, psychotherapy coverage pathways.
- location_maps  : Directions, public transit, "how do I get to", opening hours, addresses.
- migrant_health : Support specific to migrants, refugees, asylum seekers, new arrivals,
                   interpreters, uninsured care, integration into the health system.

Respond with ONLY valid JSON: {{"intent": "<one_of_above>", "confidence": <0.0-1.0>}}
No explanation, no markdown, just the JSON.

Message: {message}
"""


async def run_general_purpose_agent(state: MedBotState) -> MedBotState:
    """Orchestrator node: pick the Level-3 sub-intent and record it in state."""
    user_input = state.get("user_input", "")

    # ── Rule-based fast-path (restricted to the four sub-intents) ──────────
    rule_intent = rule_based_intent(user_input)
    if rule_intent in GENERAL_SUBINTENTS:
        intent, confidence = rule_intent, 0.85
        logger.info(f"[GeneralPurpose] Rule-based sub-intent: {intent}")
    else:
        # ── LLM classification among the four sub-agents ──────────────────
        try:
            llm = get_llm(temperature=0.0)
            resp = await llm.ainvoke([
                SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT.format(message=user_input)),
            ])
            raw = resp.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                intent = parsed.get("intent", DEFAULT_SUBINTENT)
                confidence = float(parsed.get("confidence", 0.7))
            else:
                intent, confidence = DEFAULT_SUBINTENT, 0.5
        except Exception as e:
            logger.error(f"General purpose orchestrator LLM error: {e}")
            intent, confidence = DEFAULT_SUBINTENT, 0.5

        if intent not in GENERAL_SUBINTENTS:
            intent = DEFAULT_SUBINTENT
        logger.info(f"[GeneralPurpose] LLM sub-intent: {intent} ({confidence:.2f})")

    return {
        **state,
        "detected_intent": intent,
        "intent_confidence": confidence,
        "is_emergency": False,
    }


def route_to_subagent(state: MedBotState) -> str:
    """Level-3 conditional edge: orchestrator → sub-agent node."""
    intent = state.get("detected_intent", DEFAULT_SUBINTENT)
    return GENERAL_SUBINTENTS.get(intent, GENERAL_SUBINTENTS[DEFAULT_SUBINTENT])
