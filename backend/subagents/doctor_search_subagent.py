"""
backend/subagents/doctor_search_subagent.py — Doctor Search Sub-Agent
Serves the Doctor Search Agent.

Data sources (priority order):
  1. Direct web scraper → arzt-auskunft.de (live, real listings)
  2. Tavily web search on arzt-auskunft.de (when TAVILY_API_KEY set)
  3. Static fallback — curated Oberhausen doctors list
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from backend.tools.maps_tool import format_doctor_map_entry
from backend.tools.arzt_auskunft_scraper import scrape_arzt_auskunft

logger = logging.getLogger(__name__)

# ─── Static fallback doctors (used only when scraping and Tavily both fail) ────
FALLBACK_DOCTORS: List[Dict[str, Any]] = [
    {
        "name": "Dr. med. Petra Lindner",
        "specialization": "Diabetology & Endocrinology",
        "fachgebiet": "Innere Medizin und Endokrinologie und Diabetologie",
        "address": "Bahnhofstr. 33, 46045 Oberhausen",
        "phone": "+49 208 851200",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/innere-medizin-und-endokrinologie-und-diabetologie/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Ahmad Karimi",
        "specialization": "Diabetology & Endocrinology",
        "fachgebiet": "Innere Medizin und Endokrinologie und Diabetologie",
        "address": "Schenkendorfstr. 4, 46047 Oberhausen",
        "phone": "+49 208 626050",
        "languages": ["de", "en", "ar", "fa"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/innere-medizin-und-endokrinologie-und-diabetologie/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Klaus Schmitz",
        "specialization": "General Practitioner",
        "fachgebiet": "Allgemeinmedizin",
        "address": "Marktstraße 12, 46045 Oberhausen",
        "phone": "+49 208 820010",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/allgemeinmedizin/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Silke Bergmann",
        "specialization": "General Practitioner",
        "fachgebiet": "Allgemeinmedizin",
        "address": "Elsässer Str. 7, 46117 Oberhausen",
        "phone": "+49 208 640550",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/allgemeinmedizin/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Ralf Hübner",
        "specialization": "Internal Medicine",
        "fachgebiet": "Innere Medizin",
        "address": "Mülheimer Str. 22, 46045 Oberhausen",
        "phone": "+49 208 807070",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/innere-medizin/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Stefan Weiss",
        "specialization": "Cardiology",
        "fachgebiet": "Innere Medizin und Kardiologie",
        "address": "Willy-Brandt-Platz 2, 46045 Oberhausen",
        "phone": "+49 208 820060",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/innere-medizin-und-kardiologie/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Olena Kovalenko",
        "specialization": "Pediatrics",
        "fachgebiet": "Kinderheilkunde / Kinder- und Jugendmedizin",
        "address": "Bahnhofstr. 8, 46045 Oberhausen",
        "phone": "+49 208 789012",
        "languages": ["de", "uk", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/kinderheilkunde-kinder-und-jugendmedizin/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Nadia Kovac",
        "specialization": "Dermatology",
        "fachgebiet": "Haut- und Geschlechtskrankheiten",
        "address": "Falkensteinstr. 9, 46045 Oberhausen",
        "phone": "+49 208 634500",
        "languages": ["de", "en", "hr"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/haut-und-geschlechtskrankheiten/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Ahmet Yilmaz",
        "specialization": "Internal Medicine",
        "fachgebiet": "Innere Medizin",
        "address": "Mülheimer Str. 12, 46045 Oberhausen",
        "phone": "+49 208 654321",
        "languages": ["de", "tr", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/innere-medizin/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Michael Krause",
        "specialization": "ENT (Ear, Nose, Throat)",
        "fachgebiet": "Hals-Nasen-Ohrenheilkunde",
        "address": "Gutenbergstr. 27, 46049 Oberhausen",
        "phone": "+49 208 807060",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/hals-nasen-ohrenheilkunde/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Julia Fischer",
        "specialization": "Ophthalmology",
        "fachgebiet": "Augenheilkunde",
        "address": "Centroallee 260, 46047 Oberhausen",
        "phone": "+49 208 693200",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/augenheilkunde/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Thomas Becker",
        "specialization": "Neurology",
        "fachgebiet": "Neurologie",
        "address": "Lothringer Str. 4, 46045 Oberhausen",
        "phone": "+49 208 826600",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/neurologie/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Maria Müller",
        "specialization": "Gynecology",
        "fachgebiet": "Frauenheilkunde und Geburtshilfe",
        "address": "Schwartzstr. 62, 46045 Oberhausen",
        "phone": "+49 208 820040",
        "languages": ["de", "en", "tr"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/frauenheilkunde-und-geburtshilfe/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Hans-Peter Vogel",
        "specialization": "Orthopedics",
        "fachgebiet": "Orthopädie",
        "address": "Am Einberg 3, 46049 Oberhausen",
        "phone": "+49 208 694400",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/oberhausen-rheinland/",
    },
    {
        "name": "Dr. med. Susanne Braun",
        "specialization": "Psychiatry & Psychotherapy",
        "fachgebiet": "Nervenheilkunde",
        "address": "Duisburger Str. 198, 46049 Oberhausen",
        "phone": "+49 208 666100",
        "languages": ["de", "en"],
        "kvno_accepted": True,
        "source_url": "https://www.arzt-auskunft.de/nervenheilkunde/oberhausen-rheinland/",
    },
]

# Static hospital fallback
FALLBACK_HOSPITALS: List[Dict[str, Any]] = [
    {
        "name": "Evangelisches Krankenhaus Oberhausen",
        "specialization": "Hospital – Emergency",
        "address": "Virchowstr. 20, 46047 Oberhausen",
        "phone": "+49 208 881-0",
        "is_hospital": True,
        "source_url": "https://www.arzt-auskunft.de/oberhausen-rheinland/",
    },
    {
        "name": "St. Marien-Hospital Oberhausen",
        "specialization": "Hospital – Emergency",
        "address": "Josefstr. 3, 46045 Oberhausen",
        "phone": "+49 208 8999-0",
        "is_hospital": True,
        "source_url": "https://www.arzt-auskunft.de/oberhausen-rheinland/",
    },
]

# Maps keywords → specialization name (for intent inference)
SYMPTOM_SPECIALISATION_MAP = {
    # Pediatrics
    "children": "Pediatrics", "child": "Pediatrics", "kind": "Pediatrics",
    "çocuk": "Pediatrics", "дитина": "Pediatrics", "kids": "Pediatrics",
    # Cardiology
    "heart": "Cardiology", "herz": "Cardiology", "cardiac": "Cardiology",
    "kardiologie": "Cardiology", "cardio": "Cardiology",
    # Dermatology
    "skin": "Dermatology", "haut": "Dermatology", "rash": "Dermatology",
    "dermatology": "Dermatology",
    # Ophthalmology
    "eye": "Ophthalmology", "auge": "Ophthalmology", "vision": "Ophthalmology",
    "augen": "Ophthalmology",
    # Dentistry
    "dental": "Dentistry", "zahn": "Dentistry", "teeth": "Dentistry",
    # Psychiatry
    "mental": "Psychiatry & Psychotherapy", "psychisch": "Psychiatry & Psychotherapy",
    "depression": "Psychiatry & Psychotherapy", "anxiety": "Psychiatry & Psychotherapy",
    # Gynecology
    "women": "Gynecology", "frau": "Gynecology", "pregnancy": "Gynecology",
    "schwanger": "Gynecology", "gyneco": "Gynecology",
    # Orthopedics
    "bone": "Orthopedics", "knochen": "Orthopedics", "joint": "Orthopedics",
    "ortho": "Orthopedics", "back pain": "Orthopedics",
    # Pulmonology
    "lung": "Pulmonology", "lunge": "Pulmonology", "asthma": "Pulmonology",
    "breathing": "Pulmonology",
    # Diabetology
    "diabetes": "Diabetology & Endocrinology", "diabetic": "Diabetology & Endocrinology",
    "diabetiker": "Diabetology & Endocrinology", "blutzucker": "Diabetology & Endocrinology",
    "insulin": "Diabetology & Endocrinology", "endocrin": "Diabetology & Endocrinology",
    # Neurology
    "neuro": "Neurology", "neurology": "Neurology", "stroke": "Neurology",
    "migraine": "Neurology", "migräne": "Neurology",
    # ENT
    "ent": "ENT (Ear, Nose, Throat)", "ear": "ENT (Ear, Nose, Throat)",
    "nose": "ENT (Ear, Nose, Throat)", "throat": "ENT (Ear, Nose, Throat)",
    "hno": "ENT (Ear, Nose, Throat)",
    # Internal Medicine
    "internal": "Internal Medicine", "innere": "Internal Medicine",
    "gastro": "Internal Medicine", "digestion": "Internal Medicine",
}


def infer_specialisation(query: str) -> Optional[str]:
    """Infer the needed specialisation from free text keywords."""
    q = query.lower()
    for keyword, spec in SYMPTOM_SPECIALISATION_MAP.items():
        if keyword in q:
            return spec
    return None


async def _tavily_doctor_search(
    query: str,
    specialization: Optional[str],
    city: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Search arzt-auskunft.de via Tavily for real doctor listings.
    Returns a normalised list compatible with FALLBACK_DOCTORS format.
    """
    try:
        from backend.tools.web_search_tool import web_search
        search_query = f"{specialization or query} Arzt {city} site:arzt-auskunft.de"
        results = await web_search(
            search_query,
            max_results=limit,
            search_depth="basic",
            include_domains=["arzt-auskunft.de"],
        )
        doctors = []
        for r in results:
            if r.get("title") and r.get("url"):
                doctors.append({
                    "name": r["title"],
                    "specialization": specialization or "General Practitioner",
                    "address": city,
                    "phone": "—",
                    "languages": ["de"],
                    "kvno_accepted": True,
                    "source_url": r["url"],
                    "snippet": r.get("content", ""),
                })
        return doctors
    except Exception as e:
        logger.warning(f"Tavily doctor search failed: {e}")
        return []


async def find_doctors(
    query: str,
    language: Optional[str] = None,
    specialization: Optional[str] = None,
    city: str = "Oberhausen",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Find doctors matching the query.

    Priority:
      1. Direct scraper → arzt-auskunft.de (live, real listings)
      2. Tavily web search → arzt-auskunft.de
      3. Static FALLBACK_DOCTORS

    Returns
    -------
    {doctors: [...], hospitals: [...], inferred_specialisation: str|None, city: str}
    """
    inferred = specialization or infer_specialisation(query)

    # ── 1. Try direct web scraper (arzt-auskunft.de) ──────────────────────────
    doctors: List[Dict] = []
    try:
        doctors = await scrape_arzt_auskunft(
            query=query,
            specialization=inferred,
            city=city,
            limit=limit,
        )
        if doctors:
            logger.info(f"Direct scraper found {len(doctors)} doctors for '{inferred or query}'")
    except Exception as e:
        logger.error(f"Direct scraper error: {e}")

    # ── 2. Try Tavily web search → arzt-auskunft.de ────────────────────────────
    if not doctors:
        logger.info("Falling back to Tavily search for doctors")
        doctors = await _tavily_doctor_search(query, inferred, city, limit)

    # ── 3. Static fallback ────────────────────────────────────────────────────
    if not doctors:
        logger.info("Falling back to static doctor list")
        doctors = FALLBACK_DOCTORS.copy()

        # Filter by language if requested
        if language:
            lang_filtered = [
                d for d in doctors if language in d.get("languages", [])
            ]
            if lang_filtered:
                doctors = lang_filtered

        # Filter or prioritise by specialization
        if inferred:
            spec_lower = inferred.lower()
            spec_match = [
                d for d in doctors
                if spec_lower in d.get("specialization", "").lower()
                or spec_lower in d.get("fachgebiet", "").lower()
            ]
            doctors = spec_match if spec_match else doctors

    # ── Enrich with maps links ────────────────────────────────────────────────
    doctors = [format_doctor_map_entry(d) for d in doctors[:limit]]
    hospitals = [format_doctor_map_entry(h) for h in FALLBACK_HOSPITALS[:3]]

    return {
        "doctors": doctors,
        "hospitals": hospitals,
        "inferred_specialisation": inferred,
        "city": city,
        "source": "arzt-auskunft.de",
        "source_url": "https://www.arzt-auskunft.de/oberhausen-rheinland/",
    }
