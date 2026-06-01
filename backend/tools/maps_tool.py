"""
backend/tools/maps_tool.py — Maps and location services
Uses Google Maps API if key available, otherwise OpenStreetMap/Nominatim
"""
from __future__ import annotations
import logging
import httpx
from typing import Dict, Any, Optional, List, Tuple

from backend.config import settings

logger = logging.getLogger(__name__)

OBERHAUSEN_CENTER = (51.4696, 6.8630)   # lat, lng


async def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """Convert address string to (lat, lng) using Nominatim (free, no key needed)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": address, "format": "json", "limit": 1},
                headers={"User-Agent": "P4H-MedBot/1.0"},
            )
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
    return None


def google_maps_directions_url(
    destination: str,
    origin: str = "Oberhausen, Germany",
    mode: str = "transit",
) -> str:
    """Build a Google Maps directions URL (works without API key)."""
    from urllib.parse import quote
    base = "https://www.google.com/maps/dir/"
    return f"{base}{quote(origin)}/{quote(destination)}?travelmode={mode}"


def openstreetmap_url(lat: float, lng: float, zoom: int = 16) -> str:
    """Build an OpenStreetMap link."""
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map={zoom}/{lat}/{lng}"


async def get_nearby_hospitals(city: str = "Oberhausen") -> List[Dict[str, Any]]:
    """Return static list of key hospitals in Oberhausen (fallback when DB unavailable)."""
    return [
        {
            "name": "Evangelisches Krankenhaus Oberhausen",
            "address": "Virchowstr. 20, 46047 Oberhausen",
            "phone": "+49 208 881-0",
            "maps_url": google_maps_directions_url(
                "Evangelisches Krankenhaus Oberhausen, Virchowstr. 20"
            ),
            "emergency": True,
        },
        {
            "name": "St. Marien-Hospital Oberhausen",
            "address": "Josefstr. 3, 46045 Oberhausen",
            "phone": "+49 208 8999-0",
            "maps_url": google_maps_directions_url(
                "St. Marien-Hospital, Josefstr. 3 Oberhausen"
            ),
            "emergency": True,
        },
        {
            "name": "HELIOS St. Elisabeth Gruppe – Niederrhein",
            "address": "Steinbrinkstr. 96, 46145 Oberhausen",
            "phone": "+49 208 6996-0",
            "maps_url": google_maps_directions_url(
                "HELIOS St. Elisabeth Gruppe, Steinbrinkstr. 96 Oberhausen"
            ),
            "emergency": False,
        },
    ]


async def get_vvr_transit_info(origin: str, destination: str) -> Dict[str, Any]:
    """Placeholder for VRR (Verkehrsverbund Rhein-Ruhr) transit info."""
    return {
        "provider": "VRR",
        "info": "For real-time transit info, visit: https://www.vrr.de",
        "journey_planner": f"https://www.vrr.de/en/tickets-tariffs/travel-information/",
        "directions_url": google_maps_directions_url(destination, origin, "transit"),
    }


def format_doctor_map_entry(doctor: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a doctor record with map links."""
    addr = doctor.get("address", "")
    if addr:
        doctor["maps_url"] = google_maps_directions_url(addr)
        loc = doctor.get("location", {})
        if loc and "coordinates" in loc:
            lng, lat = loc["coordinates"]
            doctor["osm_url"] = openstreetmap_url(lat, lng)
    return doctor
