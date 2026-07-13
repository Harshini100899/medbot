"""
backend/tools/maps_search_tool.py — Maps & Location Retrieval Tool
Used by the Location & Maps sub-agent.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List

from backend.tools.maps_tool import (
    get_nearby_hospitals,
    get_vvr_transit_info,
    google_maps_directions_url,
    openstreetmap_url,
    OBERHAUSEN_CENTER,
)
from backend.db.mongodb import search_doctors, get_pharmacies

logger = logging.getLogger(__name__)


async def get_location_info(
    query: str,
    location_type: str = "general",
    language: str = "en",
) -> Dict[str, Any]:
    """
    Return location information based on the query type.

    location_type: "hospital" | "doctor" | "pharmacy" | "transit" | "general"
    """
    result: Dict[str, Any] = {
        "query": query,
        "location_type": location_type,
        "places": [],
        "transit": None,
        "map_center": OBERHAUSEN_CENTER,
    }

    q_lower = query.lower()

    # Detect location type from query if not specified
    if location_type == "general":
        if any(w in q_lower for w in ["hospital", "krankenhaus", "hastane", "лікарня", "emergency", "notfall"]):
            location_type = "hospital"
        elif any(w in q_lower for w in ["pharmacy", "apotheke", "eczane", "аптека", "medicine", "drug"]):
            location_type = "pharmacy"
        elif any(w in q_lower for w in ["bus", "tram", "bahn", "transit", "transport", "ubahn", "otobüs"]):
            location_type = "transit"
        elif any(w in q_lower for w in ["doctor", "arzt", "doktor", "лікар", "physician"]):
            location_type = "doctor"

    if location_type == "hospital":
        hospitals = await get_nearby_hospitals()
        result["places"] = hospitals
        result["location_type"] = "hospital"

    elif location_type == "pharmacy":
        pharmas = await get_pharmacies("Oberhausen")
        if not pharmas:
            pharmas = [
                {
                    "name": "Rathaus-Apotheke Oberhausen",
                    "address": "Marktstraße 1, 46045 Oberhausen",
                    "phone": "+49 208 200100",
                    "hours": "Mo-Fr 08:00-18:30, Sa 09:00-14:00",
                    "maps_url": google_maps_directions_url("Rathaus-Apotheke Oberhausen"),
                }
            ]
        else:
            for p in pharmas:
                if "address" in p:
                    p["maps_url"] = google_maps_directions_url(p["address"])
        result["places"] = pharmas

    elif location_type == "doctor":
        doctors = await search_doctors(city="Oberhausen", limit=3)
        for d in doctors:
            if "address" in d:
                d["maps_url"] = google_maps_directions_url(d["address"])
        result["places"] = doctors

    elif location_type == "transit":
        result["transit"] = await get_vvr_transit_info("current location", "Oberhausen")

    return result


def format_places_for_response(places: List[Dict], language: str = "en") -> str:
    """Format a list of places into readable text."""
    if not places:
        return ""
    lines = []
    for p in places:
        name = p.get("name", "")
        addr = p.get("address", "")
        phone = p.get("phone", "")
        maps_url = p.get("maps_url", "")
        line = f"📍 **{name}**"
        if addr:
            line += f"\n   📌 {addr}"
        if phone:
            line += f"\n   📞 {phone}"
        if maps_url:
            line += f"\n   🗺️ [Directions]({maps_url})"
        lines.append(line)
    return "\n\n".join(lines)
