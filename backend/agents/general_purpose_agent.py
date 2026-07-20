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
from backend.agents.supervisor_agent import rule_based_subintents, MAX_GENERAL_INTENTS as MAX_INTENTS
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

DEFAULT_SUBINTENT = "doctor_search"

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for the general-purpose branch of a
medical assistant in Oberhausen, Germany. Route the user's message to ONE, TWO, or THREE
sub-agents (use more than one only if the message genuinely spans multiple domains, e.g. "does
insurance cover this AND find me a therapist", or a refugee-specific compound question):

- doctor_search  : Find a doctor, specialist, clinic, hospital or pharmacy; book an appointment.
- policy_rights  : Health insurance (GKV/PKV), asylum health rights, prescriptions, coverage,
                   administrative/legal healthcare questions, psychotherapy coverage pathways.
- location_maps  : Directions, public transit, "how do I get to", opening hours, addresses.
- migrant_health : Support specific to migrants, refugees, asylum seekers, new arrivals,
                   interpreters, uninsured care, integration into the health system.

If the message mentions being a refugee/migrant/asylum seeker AND also asks a general
insurance/rights question, include BOTH migrant_health and policy_rights (there's room for
both alongside doctor_search) rather than picking just one.

EXAMPLES:
Message: "I am depressed and anxious. Does GKV cover psychotherapy, and find a therapist in Oberhausen?"
Output: {{"intents": ["doctor_search", "policy_rights"], "confidence": 0.9}}

Message: "I'm a refugee with chronic anxiety and no insurance -- what are my healthcare rights, does GKV cover psychotherapy for refugees, and can you find me a therapist nearby?"
Output: {{"intents": ["doctor_search", "migrant_health", "policy_rights"], "confidence": 0.9}}

Message: "How do I get to the nearest pharmacy by tram?"
Output: {{"intents": ["location_maps"], "confidence": 0.85}}

Respond with ONLY valid JSON: {{"intents": ["<one_to_three_of_above>"], "confidence": <0.0-1.0>}}
No explanation, no markdown, just the JSON.

Message: {message}
"""


async def _classify_via_llm(user_input: str) -> tuple[list[str], float]:
    """LLM classification among the four sub-agents. Returns (intents, confidence)."""
    try:
        llm = get_llm(temperature=0.0)
        resp = await llm.ainvoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT.format(message=user_input)),
        ])
        raw = resp.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            intents = parsed.get("intents", [DEFAULT_SUBINTENT])
            confidence = float(parsed.get("confidence", 0.7))
        else:
            intents, confidence = [DEFAULT_SUBINTENT], 0.5
    except Exception as e:
        logger.error(f"General purpose orchestrator LLM error: {e}")
        intents, confidence = [DEFAULT_SUBINTENT], 0.5

    intents = [i for i in intents if i in GENERAL_SUBINTENTS][:MAX_INTENTS]
    return intents, confidence


def _apply_migrant_health_preference(intents: list[str]) -> list[str]:
    """migrant_health is a specialised superset of policy_rights for the refugee/
    migrant persona -- only drop the generic policy_rights_agent when keeping
    both would squeeze something out of the cap (mirrors the same conditional
    preference in rule_based_subintents; needed here too since merging with the
    LLM's suggestions can reintroduce the pair)."""
    if (
        "migrant_health" in intents
        and "policy_rights" in intents
        and len(intents) > MAX_INTENTS
    ):
        return [i for i in intents if i != "policy_rights"]
    return intents


async def run_general_purpose_agent(state: MedBotState) -> MedBotState:
    """Orchestrator node: pick the Level-3 sub-intent(s) and record them in state."""
    user_input = state.get("user_input", "")

    # ── Rule-based fast-path (restricted to the four sub-intents) ──────────
    rule_intents = rule_based_subintents(user_input)

    if len(rule_intents) >= MAX_INTENTS:
        # Already at the cap — nothing more the LLM could add, skip the call.
        intents, confidence = rule_intents, 0.85
        logger.info(f"[GeneralPurpose] Rule-based sub-intents (at cap): {intents}")
    else:
        # Rule-based found 0 or 1 intent(s). Keyword lists can't cover every
        # specialty/phrasing/language, so always double-check with the LLM
        # (which has general language understanding + few-shot examples) and
        # merge, instead of trusting an incomplete rule-based scan as final.
        llm_intents, llm_confidence = await _classify_via_llm(user_input)

        merged = list(rule_intents)
        for i in llm_intents:
            if i not in merged:
                merged.append(i)
        merged = _apply_migrant_health_preference(merged)
        intents = merged[:MAX_INTENTS] or [DEFAULT_SUBINTENT]
        confidence = 0.85 if rule_intents else llm_confidence

        logger.info(
            f"[GeneralPurpose] Hybrid sub-intents: rule={rule_intents} llm={llm_intents} "
            f"-> merged={intents} ({confidence:.2f})"
        )

    return {
        **state,
        "detected_intent": intents[0],
        "active_intents": intents,
        "intent_confidence": confidence,
        "is_emergency": False,
    }


def route_to_subagent(state: MedBotState) -> list[str]:
    """Level-3 conditional edge: orchestrator → one or more sub-agent nodes (fan-out)."""
    intents = state.get("active_intents") or [state.get("detected_intent", DEFAULT_SUBINTENT)]
    nodes = [GENERAL_SUBINTENTS[i] for i in intents if i in GENERAL_SUBINTENTS]
    return nodes or [GENERAL_SUBINTENTS[DEFAULT_SUBINTENT]]
