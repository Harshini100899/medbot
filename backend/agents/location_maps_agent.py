"""
backend/agents/location_maps_agent.py — Location & Maps Agent
Handles directions, transit, and facility location queries
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.subagents.maps_subagent import get_location_info, format_places_for_response
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

MAPS_SYSTEM_PROMPT = """You are a helpful location and navigation assistant for Oberhausen, Germany.
{lang_instruction}

LOCATION DATA FOUND:
{location_data}

GUIDELINES:
- Provide clear directions and location information
- Mention public transit options (VRR — Verkehrsverbund Rhein-Ruhr)
- Use the VRR journey planner: https://www.vrr.de for real-time schedules
- Mention bus/tram lines in Oberhausen where relevant (e.g. Tram 105, Bus 952)
- For emergencies, always remind them 112 provides ambulance service
- Include opening hours if available
- Oberhausen Hauptbahnhof is the main transport hub
"""


async def run_location_maps_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")

    # ── Get location info ──────────────────────────────────────────────────
    loc_result = await get_location_info(user_input, language=lang)
    places = loc_result.get("places", [])
    transit = loc_result.get("transit")

    # Format for display
    places_text = format_places_for_response(places, lang)
    transit_text = ""
    if transit:
        transit_text = f"\n\n**Transit:** {transit.get('info','')} — {transit.get('journey_planner','')}"

    location_data = (places_text or "No specific facilities found.") + transit_text

    prompt = MAPS_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        location_data=location_data,
    )

    try:
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input),
        ])
        answer = resp.content.strip()
    except Exception as e:
        logger.error(f"Maps LLM error: {e}")
        answer = "Here are the locations I found:\n\n" + location_data

    sources = [
        {"type": "maps", "title": p.get("name", ""), "url": p.get("maps_url", "")}
        for p in places[:3]
    ]

    return {
        **state,
        "active_agent": "location_maps_agent",
        "agent_raw_output": answer,
        "sources": sources,
        "needs_maps": False,
        "needs_disclaimer": False,
        "is_emergency": False,
        "extra_context": {"places": places},
    }
