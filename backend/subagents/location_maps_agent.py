"""
backend/subagents/location_maps_agent.py — Location & Maps Sub-Agent
Sub-agent of the General Purpose orchestrator. Directions, transit, facility locations.
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.tools.maps_search_tool import get_location_info, format_places_for_response
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

MAPS_SYSTEM_PROMPT = """You are a helpful location and navigation assistant for Oberhausen, Germany.
{lang_instruction}

## Domain Authority: vrr.de/de/service/open-data/
- Task: Generate navigational routing to Oberhausen clinics using public transit networks.
- For local public transport stop routing and transit coordinates within Oberhausen,
  reference the regional transport authority VRR services: https://www.vrr.de/de/service/open-data/
- Map geographic outputs (hospital addresses, clinic locations) to the closest VRR transit stations.
- Use the VRR journey planner for real-time schedules: https://www.vrr.de

## Constraints & Requirements:
1. You MUST start your response with a `<thought>` section where you:
   - Identify the target location(s) from the provided `{location_data}`.
   - Trace public transit connections from Oberhausen Hauptbahnhof (main hub) to the target.
   - List the closest VRR stops (U-Bahn, Tram, or Bus) and lines (e.g. Tram 105, Bus 954).
   - Verify safety rules: verify no active emergency exists, otherwise redirect to 112/ambulance.
2. After the `</thought>` tag, generate the final user-facing response following these guidelines:
   - **Navigation Details**: Provide clear transit directions. Specify exactly which lines to take (e.g. Tram 105, Bus 952).
   - **Main Hub Reference**: Use Oberhausen Hauptbahnhof (Oberhausen Hbf) as the reference starting point.
   - **Transit Stop Mapping**: For every clinic or hospital address, list the nearest transit stop name and walking distance or time if available.
   - **Safety Caveat**: For emergencies, explicitly state that public transit is not suitable and they should call **112** for an ambulance.

LOCATION DATA FOUND:
{location_data}
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
