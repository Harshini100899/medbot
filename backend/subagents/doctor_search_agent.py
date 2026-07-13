"""
backend/subagents/doctor_search_agent.py — Doctor Search Sub-Agent
Sub-agent of the General Purpose orchestrator. Finds healthcare providers in Oberhausen.
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.tools.doctor_search_tool import find_doctors
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

DOCTOR_SYSTEM_PROMPT = """You are a professional, empathetic healthcare navigator representing MedBot in Oberhausen, Germany.
{lang_instruction}

Your task is to help the user find the most appropriate healthcare provider or doctor in Oberhausen based on the live search listings provided below.

AVAILABLE DOCTORS / FACILITIES (LATEST LIVE WEB SEARCH DATA):
{doctor_info}

GUIDELINES FOR RECOMMENDATION:
1. **Live Search Priority**: You must only present and discuss doctors from the live web search data list above. Do NOT invent names, addresses, or phone numbers. If no doctors are listed, explain that you couldn't find any listings for this specialty in Oberhausen right now, and suggest searching the directories jameda.de or arzt-auskunft.de.
2. **Languages & GKV/PKV**: Pay close attention to the languages spoken by each doctor in the list and point out those that match the user's preference. Mention if they accept GKV (statutory health insurance / 'Kassenpatienten').
3. **Booking Appointments**: Explain how to schedule appointments. German doctor surgeries ('Praxen') usually require booking in advance by calling them, or through online platforms like jameda.de or doctolib.de.
4. **General Practitioner ('Hausarzt') Rule**: Remind the user that for general or new symptoms, they should first visit a General Practitioner (Hausarzt) who can provide a referral ('Überweisung') to specialists if needed.
5. **After-Hours Care**: If the request is for after-hours or weekend medical assistance that is NOT a life-threatening emergency, instruct them to call the national doctor-on-call service at **116 117** (Ärztlicher Bereitschaftsdienst) or visit the nearest standby practice ('Bereitschaftspraxis').
6. **Tone & Empathy**: Maintain a warm, welcoming, and reassuring tone. Many users are newcomers, refugees, or non-native speakers who find the German medical bureaucracy overwhelming. Be extremely clear and step-by-step.

Inferred specialisation needed: {inferred_spec}
"""


def _format_doctor_list(doctors: list, hospitals: list) -> str:
    parts = []
    if doctors:
        parts.append("DOCTORS:")
        for d in doctors:
            entry = (
                f"  - {d.get('name','N/A')} | {d.get('specialization','N/A')}\n"
                f"    Address: {d.get('address','N/A')}\n"
                f"    Phone: {d.get('phone','N/A')}\n"
                f"    Languages: {', '.join(d.get('languages', []))}\n"
                f"    KVNO/GKV: {'Yes' if d.get('kvno_accepted') else 'No'}"
            )
            if d.get("maps_url"):
                entry += f"\n    Directions: {d['maps_url']}"
            parts.append(entry)
    if hospitals:
        parts.append("\nHOSPITALS (A&E / Emergency):")
        for h in hospitals:
            parts.append(
                f"  - {h.get('name','N/A')} | {h.get('address','N/A')} | 📞 {h.get('phone','N/A')}"
            )
    return "\n".join(parts)


async def run_doctor_search_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")

    # ── Find doctors ──────────────────────────────────────────────────────
    result = await find_doctors(query=user_input, language=lang)
    doctors = result.get("doctors", [])
    hospitals = result.get("hospitals", [])
    inferred_spec = result.get("inferred_specialisation") or "General Practitioner"

    doctor_info = _format_doctor_list(doctors, hospitals)

    # ── LLM call ──────────────────────────────────────────────────────────
    prompt = DOCTOR_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        doctor_info=doctor_info or "No specific doctors found in database.",
        inferred_spec=inferred_spec,
    )
    try:
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input),
        ])
        answer = resp.content.strip()
    except Exception as e:
        logger.error(f"Doctor search LLM error: {e}")
        answer = "I found some healthcare providers for you:\n\n" + doctor_info

    sources = [
        {"type": "database", "title": d.get("name", ""), "url": d.get("maps_url", "")}
        for d in doctors[:3]
    ]

    return {
        **state,
        "active_agent": "doctor_search_agent",
        "agent_raw_output": answer,
        "sources": sources,
        "needs_disclaimer": True,
        "needs_maps": False,
        "is_emergency": False,
        "extra_context": {"doctors": doctors, "hospitals": hospitals},
    }
