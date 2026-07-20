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
7. **Clickable Profile Links**: Every doctor listed above has a "Profile Link". You MUST format each doctor's name as a clickable markdown link to that URL, exactly like this:
   `[Herr/Frau Dr. Name](Profile Link) (Specialization) - Address: ..., Phone: ...`
   Never omit the link and never invent a URL that isn't in the data above.
8. **Response Hygiene**: Do NOT add a generic "if this is a life-threatening emergency, call 112" line — that is automatically appended once at the end of the final response, so repeating it is redundant. Do NOT end with a generic warm sign-off (e.g. "take care of yourself", "don't hesitate to reach out") — your response may be combined with another agent's answer, so stop as soon as you've given the doctor recommendations and booking info.

EXAMPLE (this is the required format — follow it exactly):
Given this entry in AVAILABLE DOCTORS / FACILITIES:
  - Herr Gerhard Bongers | Facharzt für Psychiatrie und Psychotherapie
    Address: Bahnhofstraße 64, 46145 Oberhausen-Sterkrade
    Phone: 02 0866 00 40
    Languages: de
    KVNO/GKV: Yes
    Profile Link: https://www.arzt-auskunft.de/psychiatrie-und-psychotherapie/oberhausen-rheinland/12345
You must write it as:
  [Herr Gerhard Bongers](https://www.arzt-auskunft.de/psychiatrie-und-psychotherapie/oberhausen-rheinland/12345) (Facharzt für Psychiatrie und Psychotherapie) - Address: Bahnhofstraße 64, 46145 Oberhausen-Sterkrade, Phone: 02 0866 00 40

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
                f"    KVNO/GKV: {'Yes' if d.get('kvno_accepted') else 'No'}\n"
                f"    Profile Link: {d.get('source_url', 'N/A')}"
            )
            parts.append(entry)
    if hospitals:
        parts.append("\nHOSPITALS (A&E / Emergency):")
        for h in hospitals:
            parts.append(
                f"  - {h.get('name','N/A')} | {h.get('address','N/A')} | 📞 {h.get('phone','N/A')}"
            )
    return "\n".join(parts)


def _format_doctor_list_markdown(doctors: list, hospitals: list) -> str:
    """User-facing fallback formatter (used when the LLM call itself fails) --
    unlike _format_doctor_list (which builds the raw prompt context for the
    LLM), this produces the same polished, clickable-link output the LLM is
    normally instructed to produce, so a failed LLM call doesn't dump internal
    debug-style formatting straight to the user."""
    if not doctors:
        return (
            "I couldn't find any listings for this specialty in Oberhausen right now. "
            "Try searching jameda.de or arzt-auskunft.de directly, or ask again in a moment."
        )
    lines = ["Here are some healthcare providers I found for you:\n"]
    for d in doctors:
        name = d.get("name", "N/A")
        url = d.get("source_url")
        title = f"[{name}]({url})" if url else name
        langs = ", ".join(d.get("languages", [])) or "N/A"
        gkv = "Yes" if d.get("kvno_accepted") else "No"
        phone = d.get("phone") or "N/A"
        lines.append(
            f"- {title} ({d.get('specialization', 'N/A')}) - "
            f"Address: {d.get('address', 'N/A')}, Phone: {phone} "
            f"(Languages: {langs}, GKV: {gkv})"
        )
    if hospitals:
        lines.append("\n**Hospitals (A&E / Emergency):**")
        for h in hospitals:
            lines.append(f"- {h.get('name', 'N/A')} — {h.get('address', 'N/A')} — 📞 {h.get('phone', 'N/A')}")
    return "\n".join(lines)


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
        answer = _format_doctor_list_markdown(doctors, hospitals)

    sources = [
        {"type": "web", "title": d.get("name", ""), "url": d.get("source_url", "")}
        for d in doctors[:3]
    ]

    return {
        "agent_outputs": [{
            "agent": "doctor_search_agent",
            "output": answer,
            "sources": sources,
            "needs_disclaimer": True,
            "needs_maps": False,
            "extra": {"doctors": doctors, "hospitals": hospitals},
        }],
    }
