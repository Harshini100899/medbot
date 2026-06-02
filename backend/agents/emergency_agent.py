"""
backend/agents/emergency_agent.py — Emergency Medical Agent
Handles life-threatening situations with immediate guidance and emergency numbers
"""
from __future__ import annotations
import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.tools.maps_tool import get_nearby_hospitals, google_maps_directions_url
from backend.config import settings
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

# ─── Emergency instructions per language ──────────────────────────────────────
EMERGENCY_INTRO = {
    "en": (
        "🚨 **EMERGENCY SITUATION DETECTED**\n\n"
        "**CALL 112 IMMEDIATELY** (European Emergency Number — Free, 24/7)\n\n"
    ),
    "de": (
        "🚨 **NOTFALL ERKANNT**\n\n"
        "**SOFORT 112 ANRUFEN** (Europäischer Notruf — Kostenlos, 24/7)\n\n"
    ),
    "tr": (
        "🚨 **ACİL DURUM TESPİT EDİLDİ**\n\n"
        "**HEMEN 112'Yİ ARAYIN** (Avrupa Acil Numarası — Ücretsiz, 7/24)\n\n"
    ),
    "uk": (
        "🚨 **ВИЯВЛЕНА НАДЗВИЧАЙНА СИТУАЦІЯ**\n\n"
        "**НЕГАЙНО ЗАТЕЛЕФОНУЙТЕ 112** (Загальноєвропейський номер екстреної допомоги)\n\n"
    ),
}

EMERGENCY_NUMBERS_BLOCK = {
    "en": """
**Emergency Numbers (Oberhausen / Germany):**
| Service | Number |
|---------|--------|
| 🚑 Emergency / Ambulance | **112** |
| 👮 Police | **110** |
| 🏥 Doctor on Call (Ärztlicher Bereitschaftsdienst) | **116 117** |
| ☠️ Poison Control (Vergiftungsnotruf NRW) | **0228 19240** |
| 🧠 Mental Health Crisis (Telefonseelsorge) | **0800 111 0 111** |
| 👧 Youth Crisis | **0800 111 0 333** |
""",
    "de": """
**Notfallnummern (Oberhausen / Deutschland):**
| Dienst | Nummer |
|--------|--------|
| 🚑 Notruf / Krankenwagen | **112** |
| 👮 Polizei | **110** |
| 🏥 Ärztlicher Bereitschaftsdienst | **116 117** |
| ☠️ Vergiftungsnotruf NRW | **0228 19240** |
| 🧠 Telefonseelsorge | **0800 111 0 111** |
| 👧 Jugendnotfall | **0800 111 0 333** |
""",
    "tr": """
**Acil Numaralar (Oberhausen / Almanya):**
| Hizmet | Numara |
|--------|--------|
| 🚑 Acil / Ambulans | **112** |
| 👮 Polis | **110** |
| 🏥 Nöbetçi Doktor | **116 117** |
| ☠️ Zehir Danışma | **0228 19240** |
| 🧠 Ruh Sağlığı Krizi | **0800 111 0 111** |
""",
    "uk": """
**Номери екстреної допомоги (Оберхаузен / Німеччина):**
| Служба | Номер |
|--------|-------|
| 🚑 Екстрена допомога / Швидка | **112** |
| 👮 Поліція | **110** |
| 🏥 Черговий лікар | **116 117** |
| ☠️ Центр отруєнь | **0228 19240** |
| 🧠 Психологічна допомога | **0800 111 0 111** |
""",
}

EMERGENCY_SYSTEM_PROMPT = """You are a highly efficient, calm, and professional emergency medical guidance assistant.
The user or someone near them may be in a life-threatening situation.
{lang_instruction}

INSTRUCTIONS FOR EMERGENCY SITUATIONS:
1. **Absolute Priority**: Begin your response immediately by instructing the user to call **112** (European emergency services number) or **110** (Police) if not already done. Use bold text for emergency numbers.
2. **Immediate First-Aid Guidance**: If the user indicates symptoms of cardiac arrest, choking, severe bleeding, or unconsciousness, provide short, numbered, step-by-step first-aid actions (e.g. CPR sequence, recovery position, or applying pressure to a wound).
3. **Extreme Brevity**: Do not write long paragraphs or explanations. Use bullet points and short sentences. Every second counts.
4. **Non-Diagnostic**: Do not attempt to diagnose the cause. Focus only on life-saving actions to stabilize the patient until professional emergency services arrive.
5. **Reassurance**: Reassure the user that professional help is on the way, but they must act immediately.

User situation: {user_input}
"""


async def run_emergency_agent(state: MedBotState) -> MedBotState:
    """Emergency agent node — runs when emergency is detected."""
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")

    logger.warning(f"🚨 EMERGENCY AGENT triggered for session: {state.get('session_id')}")

    # Build LLM prompt for specific first-aid guidance
    prompt = EMERGENCY_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        user_input=user_input,
    )

    llm_guidance = ""
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_input),
        ]
        resp = await llm.ainvoke(messages)
        llm_guidance = resp.content.strip()
    except Exception as e:
        logger.error(f"Emergency LLM error: {e}")
        llm_guidance = "Stay calm. Call 112 now. Help is on the way."

    # Fetch nearby hospitals
    hospitals = await get_nearby_hospitals()
    hospital_text = "\n".join(
        f"🏥 {h['name']} — {h.get('address','')} — 📞 {h.get('phone','')}"
        for h in hospitals[:2]
    )

    # Assemble full response
    response = (
        EMERGENCY_INTRO.get(lang, EMERGENCY_INTRO["en"])
        + llm_guidance
        + "\n\n"
        + EMERGENCY_NUMBERS_BLOCK.get(lang, EMERGENCY_NUMBERS_BLOCK["en"])
        + "\n\n**Nearest Emergency Hospitals:**\n"
        + hospital_text
    )

    return {
        **state,
        "active_agent": "emergency_agent",
        "agent_raw_output": response,
        "is_emergency": True,
        "needs_disclaimer": False,
        "needs_maps": False,
        "sources": [{"type": "emergency", "title": "Emergency Services Germany"}],
    }
