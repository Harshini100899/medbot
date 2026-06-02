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

# ─── Static fallback doctors (disabled) ────────────────────────────────────────
FALLBACK_DOCTORS: List[Dict[str, Any]] = []

# Static hospital fallback (disabled)
FALLBACK_HOSPITALS: List[Dict[str, Any]] = []

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


def clean_doctor_name(title: str) -> str:
    import re
    # Remove site name branding (e.g. - Arzt-Auskunft, | Jameda, etc.)
    title = re.sub(r'\s*[\-|:|\|]\s*(?:Arzt-Auskunft|Jameda|KVNO|KVWL|arztsuche).*', '', title, flags=re.IGNORECASE)
    # Remove trailing stuff like "in Oberhausen"
    title = re.sub(r'\s+in\s+Oberhausen.*', '', title, flags=re.IGNORECASE)
    # Remove leading/trailing quotes and spaces
    title = title.strip().strip('"').strip("'")
    return title


async def _tavily_doctor_search(
    query: str,
    specialization: Optional[str],
    city: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Search doctor directories via Tavily for real doctor listings.
    Uses LLM extraction with regex fallback to ensure it works offline/without Ollama.
    """
    try:
        import re
        from backend.tools.web_search_tool import web_search, DOCTOR_SEARCH_DOMAINS
        search_query = f"{specialization or query} Arzt {city}"
        results = await web_search(
            search_query,
            max_results=10,  # fetch more results to get good candidates
            search_depth="advanced",
            include_domains=DOCTOR_SEARCH_DOMAINS,
        )
        if not results:
            return []

        # Try LLM first
        doctors = []
        try:
            from backend.llm_factory import get_llm
            from langchain_core.messages import SystemMessage
            import json

            llm = get_llm(temperature=0.0)
            
            # Formulate the prompt
            search_results_str = ""
            for idx, r in enumerate(results):
                search_results_str += f"[{idx}] Title: {r['title']}\nSnippet: {r['content']}\nURL: {r['url']}\n\n"

            prompt = (
                "You are an expert system that extracts doctor listings from web search results.\n"
                f"Extract a list of doctors in {city} from the search results below.\n\n"
                "Search Results:\n"
                f"{search_results_str}\n"
                "Return ONLY a valid JSON array of objects representing the doctors. Do not include markdown formatting, backticks (like ```json), or explanations, just the JSON.\n"
                "Each object in the array MUST have the following keys:\n"
                '- "name": (string, the doctor\'s name, e.g. "Dr. med. Hans Müller")\n'
                '- "specialization": (string, e.g. "General Practitioner", "Cardiology", or the inferred/requested specialization)\n'
                '- "address": (string, the full street address with postal code if found, e.g. "Mellinghofer Straße 228, 46047 Oberhausen", otherwise just the city name)\n'
                '- "phone": (string, phone number if found, otherwise "—")\n'
                '- "languages": (list of strings, languages spoken, default to ["de"] if not specified)\n'
                '- "kvno_accepted": (boolean, True if they accept GKV/statutory health insurance or KVNO, default to True)\n'
                '- "source_url": (string, the URL of the search result from which they were extracted)\n\n'
                "If no doctors are found in the search results, return an empty array []."
            )

            resp = await llm.ainvoke([
                SystemMessage(content=prompt),
            ])
            raw = resp.content.strip()
            # Clean up potential markdown formatting wrapping the JSON
            if "```" in raw:
                raw = re.sub(r'```(?:json)?', '', raw).strip()
            
            # Try to parse
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                doctors = parsed
        except Exception as llm_err:
            logger.warning(f"LLM doctor extraction failed, falling back to regex: {llm_err}")

        # If LLM failed or returned no doctors, run regex fallback
        if not doctors:
            logger.info("Running regex fallback to parse Tavily results")
            addr_pattern = re.compile(
                r'([A-Za-zäöüßÄÖÜ\s.-]+(?:str\.|straße|weg|platz|allee)\s\d+[a-z]?,?\s*(?:46\d{3})\s+Oberhausen)',
                re.IGNORECASE
            )
            phone_pattern = re.compile(
                r'(\+49\s*\(?0?\)?\s*\d+[\s.-]*\d+[\s.-]*\d+|\b0208\s*[\s/-]*\d+\b)'
            )

            for r in results:
                title = r.get("title", "")
                snippet = r.get("content", "")
                url = r.get("url", "")
                if not title:
                    continue

                # Clean name
                doc_name = clean_doctor_name(title)
                if not doc_name or ("arzt" in doc_name.lower() and len(doc_name) < 10):
                    # Skip generic result titles
                    continue

                # Extract address
                addr_match = addr_pattern.search(snippet)
                if not addr_match:
                    addr_match = addr_pattern.search(title)
                address = addr_match.group(1).strip() if addr_match else f"{city}, Germany"

                # Extract phone
                phone_match = phone_pattern.search(snippet)
                phone = phone_match.group(1).strip() if phone_match else "—"

                doctors.append({
                    "name": doc_name,
                    "specialization": specialization or "General Practitioner",
                    "address": address,
                    "phone": phone,
                    "languages": ["de"],
                    "kvno_accepted": True,
                    "source_url": url,
                })

        return doctors[:limit]
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
      1. Tavily web search (live, real listings)
      2. Direct scraper → arzt-auskunft.de
      3. Static FALLBACK_DOCTORS (disabled)

    Returns
    -------
    {doctors: [...], hospitals: [...], inferred_specialisation: str|None, city: str}
    """
    inferred = specialization or infer_specialisation(query)

    # ── 1. Try Tavily web search ──────────────────────────────────────────────
    logger.info("Searching for doctors via Tavily web search")
    doctors = await _tavily_doctor_search(query, inferred, city, limit)

    # ── 2. Try direct web scraper (arzt-auskunft.de) if Tavily fails ──────────
    if not doctors:
        logger.info("Falling back to direct scraper for doctors")
        try:
            doctors = await scrape_arzt_auskunft(
                query=query,
                specialization=inferred,
                city=city,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Direct scraper error: {e}")

    # ── Enrich with maps links ────────────────────────────────────────────────
    doctors = [format_doctor_map_entry(d) for d in doctors[:limit]]
    hospitals = [format_doctor_map_entry(h) for h in FALLBACK_HOSPITALS[:3]]

    return {
        "doctors": doctors,
        "hospitals": hospitals,
        "inferred_specialisation": inferred,
        "city": city,
        "source": "arzt-auskunft.de / jameda.de / kvno.de",
        "source_url": "https://www.arzt-auskunft.de",
    }
