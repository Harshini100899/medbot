"""
backend/agents/doctor_search_agent.py — Doctor Search Agent
Helps users find appropriate healthcare providers in Oberhausen
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.subagents.doctor_search_subagent import find_doctors
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

DOCTOR_SYSTEM_PROMPT = """You are a helpful healthcare navigator for Oberhausen, Germany.
{lang_instruction}

The user needs help finding a doctor or healthcare provider.

AVAILABLE DOCTORS / FACILITIES:
{doctor_info}

GUIDELINES:
- Recommend the most appropriate doctor(s) based on the user's needs
- Mention if doctors speak the user's language
- Note if they accept Kassenpatienten (GKV / statutory health insurance)
- Explain how to book an appointment in Germany (usually by phone, or jameda.de)
- Mention the 116 117 doctor-on-call service for after-hours non-emergencies
- Be warm and helpful — many users may be unfamiliar with the German healthcare system
- For urgent but non-life-threatening cases, mention the Notaufnahme (A&E) option

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
        "needs_maps": True,
        "is_emergency": False,
        "extra_context": {"doctors": doctors, "hospitals": hospitals},
    }
