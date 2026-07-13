"""
backend/agents/supervisor_agent.py — Supervisor Agent (Level 1 Gateway Router)

Splits the cognitive load with a cheap binary / tertiary classification:

    emergency  → emergency fast-path
    medical    → Medical Specialist Agent (clinical knowledge)
    general    → General Purpose Agent (orchestrator over the sub-agents)

Chit-chat / greetings / unclear default to ``medical`` so the informational
specialist can greet and clarify. Only clearly municipal / logistical queries
(doctor search, insurance & rights, directions, migrant admin) go to ``general``.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState, TOP_ROUTES
from backend.language.detector import detect_language
from backend.ontology.normalizer import is_emergency
from backend.llm_factory import get_llm
from backend.memory.redis_memory import cache_language, get_cached_language

logger = logging.getLogger(__name__)

# ─── Level-1 binary classifier prompt (medical vs general) ────────────────────
TOP_ROUTE_SYSTEM_PROMPT = """You are a gateway router for a medical assistant in Oberhausen, Germany.
Classify the user's message into EXACTLY ONE of two routes:

- medical : Clinical / health-information questions — symptoms, diseases, medications,
            treatments, mental health, "what is / what should I do about ...".
            ALSO use this for greetings, small talk, or unclear messages.
- general : Municipal / logistical / administrative needs — finding a doctor, clinic,
            hospital or pharmacy; health insurance, GKV/PKV, asylum health rights,
            prescriptions; directions, transit, opening hours, addresses; migrant,
            refugee or asylum-seeker administrative support.

Respond with ONLY valid JSON: {{"route": "medical"|"general", "confidence": <0.0-1.0>}}
No explanation, no markdown, just the JSON.

Message: {message}
"""


# Rule-based fast-path (avoids LLM call for obvious cases)
KEYWORD_RULES: Dict[str, list] = {
    "emergency": [
        "112", "911", "ambulance", "notfall", "notruf", "ohnmächtig", "unconscious",
        "not breathing", "heart attack", "herzinfarkt", "stroke", "schlaganfall",
        "acil", "can't breathe", "dying", "sterbe", "ich sterbe", "help me please",
    ],
    "doctor_search": [
        # English — explicit requests
        "find doctor", "find a doctor", "find me a doctor", "find a suitable doctor",
        "find specialist", "find a specialist", "need a doctor", "need a specialist",
        "looking for a doctor", "looking for a specialist", "looking for doctor",
        "recommend a doctor", "recommend doctor", "suggest a doctor",
        "doctor for", "specialist for", "physician for", "which doctor",
        "book appointment", "book a doctor", "make appointment",
        "suitable doctor", "good doctor", "nearby doctor", "local doctor",
        "doctor in oberhausen", "doctor near", "clinic near", "clinic in",
        "pharmacy", "apotheke", "eczane",
        # English — disease/condition → doctor needed
        "find me", "suggest doctor", "see a doctor", "visit a doctor",
        "who should i see", "what doctor", "which specialist",
        # German
        "arzt finden", "arzt suchen", "arzt in oberhausen", "welcher arzt",
        "doktor suchen", "facharzt", "hausarzt", "termin",
        "arzt für", "spezialist für", "diabetologe", "diabetologin",
        "arzt empfehlen", "welche klinik", "praxis",
        # Turkish
        "doktor bul", "doktor ara", "doktor için", "doktor bulmam lazım",
        "hangi doktor",
        # Disease → specialist search triggers
        "diabetes doctor", "diabetes specialist", "diabetiker arzt",
        "heart doctor", "lung doctor", "skin doctor", "eye doctor",
        # Direct disease/condition + "doctor" or implied search context
        "i have diabetes", "i have heart", "i have a diabetes",
        "ich habe diabetes", "diabetik", "high blood pressure doctor",
        "blood sugar", "blutzucker",
        # General "find me" pattern
        "find me",
    ],
    "location_maps": [
        "how do i get to", "wie komme ich", "directions", "route",
        "opening hours", "öffnungszeiten", "address", "adresse",
        "bus", "tram", "transit", "ubahn", "entfernung", "distance",
    ],
    "policy_rights": [
        "insurance", "versicherung", "gkv", "pkv", "krankenkasse",
        "asylbewerber", "aufenthaltserlaubnis", "krankenversicherung",
        "rezept", "prescription", "sigorta", "sigorta kartı", "страхування",
        "rights", "rechte", "asylum health",
    ],
    "migrant_health": [
        "refugee", "flüchtling", "migrant", "asyl", "asylbewerber",
        "neue ankunft", "neu in deutschland", "ukraine", "türkiye",
        "integration", "aufenthaltstitel", "krankenschein",
        "мігрант", "біженець", "göçmen", "sığınmacı",
    ],
}


def rule_based_intent(text: str) -> str | None:
    """
    Fast rule-based intent classification — returns intent or None.

    Also applies heuristics:
    - If the message mentions a medical condition AND a "find doctor" trigger, route to doctor_search.
    - Prioritises emergency detection above all else.
    """
    t = text.lower()

    # Priority 1: Emergency always wins
    if any(kw in t for kw in KEYWORD_RULES["emergency"]):
        return "emergency"

    # Priority 2: Explicit keyword match
    for intent, keywords in KEYWORD_RULES.items():
        if intent == "emergency":
            continue
        if any(kw in t for kw in keywords):
            return intent

    # Priority 3: Heuristic — disease/condition + search context triggers doctor_search
    CONDITION_WORDS = {
        "diabetes", "diabetic", "diabetiker", "hypertension", "high blood pressure",
        "heart disease", "asthma", "cancer", "allergy", "allergie", "arthritis",
        "depression", "anxiety", "pain", "schmerz", "fever", "fieber",
        "illness", "condition", "krankheit", "erkrankung", "symptom",
    }
    SEARCH_TRIGGERS = {
        "find", "suitable", "recommend", "need", "looking", "search",
        "oberhausen", "arzt", "doctor", "specialist", "klinik",
    }
    has_condition = any(cw in t for cw in CONDITION_WORDS)
    has_search = any(sw in t for sw in SEARCH_TRIGGERS)
    if has_condition and has_search:
        return "doctor_search"

    return None


def rule_based_top_route(text: str) -> str | None:
    """
    Cheap Level-1 route from keyword rules — returns ``emergency`` / ``medical``
    / ``general``, or ``None`` if no rule fires (defer to the binary LLM).
    """
    intent = rule_based_intent(text)
    if intent is None:
        return None
    if intent == "emergency":
        return "emergency"
    # doctor_search / policy_rights / location_maps / migrant_health → municipal
    return "general"


async def run_supervisor(state: MedBotState) -> MedBotState:
    """
    Supervisor node (Level 1 gateway router):
    1. Detect language (Redis-cached per session)
    2. Emergency fast-path
    3. Cheap binary/tertiary classify → medical | general
    """
    user_input = state.get("user_input", "")
    session_id = state.get("session_id", "")

    # ── Language Detection ─────────────────────────────────────────────────
    cached_lang = await get_cached_language(session_id)
    if cached_lang:
        lang, lang_conf = cached_lang, 0.9
    else:
        lang, lang_conf = detect_language(user_input)
        await cache_language(session_id, lang)

    # ── Emergency fast-path ────────────────────────────────────────────────
    if is_emergency(user_input, lang):
        logger.info(f"[Supervisor] Emergency fast-path | session {session_id}")
        return {
            **state,
            "user_language": lang,
            "language_confidence": lang_conf,
            "top_level_route": "emergency",
            "detected_intent": "emergency",
            "intent_confidence": 0.99,
            "is_emergency": True,
        }

    # ── Rule-based top route (fast, no LLM call) ───────────────────────────
    route = rule_based_top_route(user_input)
    confidence = 0.85

    # ── Binary LLM fallback (medical vs general) ───────────────────────────
    if route is None:
        try:
            llm = get_llm(temperature=0.0)
            resp = await llm.ainvoke([
                SystemMessage(content=TOP_ROUTE_SYSTEM_PROMPT.format(message=user_input)),
            ])
            raw = resp.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                route = parsed.get("route", "medical")
                confidence = float(parsed.get("confidence", 0.7))
            else:
                route, confidence = "medical", 0.5
        except Exception as e:
            logger.error(f"Supervisor LLM error: {e}")
            route, confidence = "medical", 0.5

    if route not in TOP_ROUTES:
        route = "medical"

    logger.info(f"[Supervisor] Route: {route} ({confidence:.2f}) | Lang: {lang} | Session: {session_id}")

    # detected_intent is refined by the General Purpose orchestrator on the
    # "general" branch; set a meaningful value now for the "medical" branch.
    detected_intent = "medical_knowledge" if route == "medical" else "general"

    return {
        **state,
        "user_language": lang,
        "language_confidence": lang_conf,
        "top_level_route": route,
        "detected_intent": detected_intent,
        "intent_confidence": confidence,
        "is_emergency": False,
    }


def route_to_agent(state: MedBotState) -> str:
    """Level-1 conditional edge: supervisor → specialist node."""
    route = state.get("top_level_route", "medical")
    return TOP_ROUTES.get(route, "medical_specialist")
