"""
backend/agents/supervisor_agent.py — Supervisor Agent
Classifies intent and routes to the appropriate specialist agent
"""
from __future__ import annotations
import json
import logging
import re
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState, INTENTS
from backend.language.detector import detect_language
from backend.ontology.normalizer import is_emergency
from backend.llm_factory import get_llm
from backend.memory.redis_memory import get_messages, cache_language, get_cached_language

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are a medical triage router. Classify the user's message into EXACTLY ONE intent.

INTENTS:
- emergency       : Life-threatening situations, cardiac arrest, choking, severe bleeding, suicidal ideation
- doctor_search   : Looking for a doctor, specialist, clinic, hospital, pharmacy
- medical_knowledge : General medical questions, symptoms, medications, diseases, treatments
- policy_rights   : Insurance questions, asylum health rights, GKV/PKV, prescriptions, legal/administrative
- location_maps   : Directions to facilities, transit, "how do I get to", opening hours, addresses
- migrant_health  : Questions specific to migrants, refugees, asylum seekers, new arrivals, integration
- general         : Greetings, unclear, or general chat not fitting above categories

Respond with ONLY valid JSON: {"intent": "<one_of_above>", "confidence": <0.0-1.0>}
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


async def run_supervisor(state: MedBotState) -> MedBotState:
    """
    Supervisor node:
    1. Detect language
    2. Check for emergency (fast path)
    3. Classify intent (rule-based → LLM)
    4. Update state for routing
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
        logger.info(f"[Supervisor] Emergency fast-path for session {session_id}")
        return {
            **state,
            "user_language": lang,
            "language_confidence": lang_conf,
            "detected_intent": "emergency",
            "intent_confidence": 0.99,
            "is_emergency": True,
        }

    # ── Rule-based intent (fast, no LLM call) ─────────────────────────────
    rule_intent = rule_based_intent(user_input)
    if rule_intent and rule_intent != "emergency":
        logger.info(f"[Supervisor] Rule-based intent: {rule_intent}")
        return {
            **state,
            "user_language": lang,
            "language_confidence": lang_conf,
            "detected_intent": rule_intent,
            "intent_confidence": 0.85,
            "is_emergency": False,
        }

    # ── LLM intent classification ──────────────────────────────────────────
    try:
        llm = get_llm(temperature=0.0)
        resp = await llm.ainvoke([
            SystemMessage(content=INTENT_SYSTEM_PROMPT.format(message=user_input)),
        ])
        raw = resp.content.strip()
        # Extract JSON even if wrapped in markdown
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            intent = parsed.get("intent", "general")
            confidence = float(parsed.get("confidence", 0.7))
        else:
            intent = "general"
            confidence = 0.5
    except Exception as e:
        logger.error(f"Supervisor LLM error: {e}")
        intent = "general"
        confidence = 0.5

    # Ensure intent is valid
    if intent not in INTENTS:
        intent = "general"

    logger.info(f"[Supervisor] Intent: {intent} ({confidence:.2f}) | Lang: {lang} | Session: {session_id}")

    return {
        **state,
        "user_language": lang,
        "language_confidence": lang_conf,
        "detected_intent": intent,
        "intent_confidence": confidence,
        "is_emergency": intent == "emergency",
    }


def route_to_agent(state: MedBotState) -> str:
    """Conditional edge routing function for LangGraph."""
    intent = state.get("detected_intent", "general")
    return INTENTS.get(intent, "medical_knowledge_agent")
