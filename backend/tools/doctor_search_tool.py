"""
backend/tools/doctor_search_tool.py — Doctor Search Retrieval Tool
Used by the Doctor Search sub-agent (and inline by other agents).

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

# ─── Single source of truth for doctor specialties ─────────────────────────
# Consolidates what used to be three independently-drifting dicts:
#   - keyword -> specialty name        (was SYMPTOM_SPECIALISATION_MAP here)
#   - specialty name -> arzt-auskunft.de URL slug (was MAPPED_KEYWORDS in
#     arzt_auskunft_scraper.py -- that file no longer has ANY specialty
#     knowledge of its own, so it can't silently fall out of sync again)
#   - specialty name -> German search term (was SPECIALISATION_GERMAN_MAP)
#
# Entry order matters: infer_specialisation() is first-match-wins, so entries
# whose keywords are substrings of a more specific term's keywords (e.g.
# "therapy" inside "physiotherapy") MUST be ordered after the more specific
# entry, or every physio query gets misrouted to psychiatry. See inline notes.
SPECIALTY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Pediatrics": {
        "keywords": ["children", "child", "kind", "çocuk", "дитина", "kids"],
        "arzt_auskunft_slug": "kinderheilkunde-kinder-und-jugendmedizin",
        "german_search_term": "Kinderarzt Kinderheilkunde",
    },
    "Cardiology": {
        "keywords": ["heart", "herz", "cardiac", "kardiologie", "cardio"],
        "arzt_auskunft_slug": "innere-medizin-und-kardiologie",
        "german_search_term": "Kardiologe Kardiologie",
    },
    "Dermatology": {
        # "dermatolog" (not the full word "dermatology") so it matches both
        # "dermatology" and "dermatologist" as a substring.
        "keywords": ["skin", "haut", "rash", "dermatolog"],
        "arzt_auskunft_slug": "haut-und-geschlechtskrankheiten",
        "german_search_term": "Hautarzt Dermatologe Dermatologie",
    },
    "Ophthalmology": {
        "keywords": ["eye", "auge", "vision", "augen", "ophthalmolog"],
        "arzt_auskunft_slug": "augenheilkunde",
        "german_search_term": "Augenarzt Augenheilkunde",
    },
    "Dentistry": {
        "keywords": ["dental", "zahn", "teeth", "dentist"],
        "arzt_auskunft_slug": "zahnmedizin",
        "german_search_term": "Zahnarzt",
    },
    # Physiotherapy — MUST come before Psychiatry below: "physiotherapy"/
    # "physiotherapist"/German "Physiotherapeut" all contain "therapy"/
    # "therapist"/"therapeut" as substrings, so this has to win the
    # first-match-wins scan or every physio query gets misrouted to psychiatry.
    # arzt_auskunft_slug is None: physiotherapists aren't "Ärzte" (physicians),
    # so this physician-only directory has no category for them at all --
    # find_doctors() falls back to the general city page + text filtering.
    "Physiotherapy": {
        "keywords": ["physiotherap", "physio", "krankengymnastik"],
        "arzt_auskunft_slug": None,
        "german_search_term": "Physiotherapeut Krankengymnastik",
    },
    "Psychiatry & Psychotherapy": {
        "keywords": [
            "mental", "psychisch", "depression", "depressed", "anxiety", "anxious",
            "therapist", "therapy", "psychotherapist", "psychiatrist", "psychologist",
            "counselor", "counsellor", "psychiater", "psychotherapeut", "therapeut",
            "psychiatrie", "psychotherapy", "psychiatr", "psych",
        ],
        "arzt_auskunft_slug": "psychiatrie-und-psychotherapie",
        "german_search_term": "Psychotherapeut Psychiater Psychotherapie",
    },
    "Gynecology": {
        "keywords": ["women", "frau", "pregnancy", "schwanger", "gyneco"],
        "arzt_auskunft_slug": "frauenheilkunde-und-geburtshilfe",
        "german_search_term": "Frauenarzt Gynäkologe Gynäkologie",
    },
    "Orthopedics": {
        "keywords": ["bone", "knochen", "joint", "ortho", "orthoped", "back pain"],
        "arzt_auskunft_slug": "orthopaedie",
        "german_search_term": "Orthopäde Orthopädie",
    },
    "Pulmonology": {
        "keywords": ["lung", "lunge", "asthma", "breathing"],
        "arzt_auskunft_slug": None,  # no dedicated arzt-auskunft.de category
        "german_search_term": "Lungenarzt Pneumologe Pneumologie",
    },
    "Diabetology & Endocrinology": {
        "keywords": ["diabetes", "diabetic", "diabetiker", "diabetolog", "blutzucker", "insulin", "endocrin"],
        "arzt_auskunft_slug": "innere-medizin-und-endokrinologie-und-diabetologie",
        "german_search_term": "Diabetologe Diabetes",
    },
    "Neurology": {
        "keywords": ["neuro", "neurology", "stroke", "migraine", "migräne", "neurolog"],
        "arzt_auskunft_slug": "neurologie",
        "german_search_term": "Neurologe Neurologie",
    },
    # ENT — bare "ent"/"ear"/"nose" deliberately avoided: they're substrings of
    # unrelated common words ("appointment", "treatment", "near", "diagnose")
    # that have nothing to do with ENT care. Use specific phrases instead.
    "ENT (Ear, Nose, Throat)": {
        "keywords": [
            "earache", "ear infection", "ear pain", "sore throat", "throat",
            "runny nose", "blocked nose", "sinus", "ent specialist", "ent doctor",
            "hno", "hals", "ohrenschmerzen", "halsschmerzen",
        ],
        "arzt_auskunft_slug": "hals-nasen-ohrenheilkunde",
        "german_search_term": "HNO Arzt Hals-Nasen-Ohrenarzt",
    },
    "Internal Medicine": {
        "keywords": ["internal", "innere", "gastro", "digestion"],
        "arzt_auskunft_slug": None,  # no dedicated arzt-auskunft.de category
        "german_search_term": "Internist Hausarzt Allgemeinmedizin",
    },
    # General Practitioner -- previously ONLY existed in arzt_auskunft_scraper.py's
    # MAPPED_KEYWORDS (mapped to a real slug) with no equivalent in
    # SYMPTOM_SPECIALISATION_MAP, so infer_specialisation("I need a GP") always
    # returned None even though the scraper had a slug ready for it. Fixed by
    # consolidation.
    "General Practitioner": {
        "keywords": ["gp", "general", "hausarzt", "allgemein"],
        "arzt_auskunft_slug": "allgemeinmedizin",
        "german_search_term": "Hausarzt Allgemeinmedizin",
    },
}


def infer_specialisation(query: str) -> Optional[str]:
    """Infer the needed specialisation from free text keywords."""
    q = query.lower()
    for specialty, entry in SPECIALTY_REGISTRY.items():
        if any(kw in q for kw in entry["keywords"]):
            return specialty
    return None


SPECIALTY_CLASSIFIER_PROMPT = """Classify this healthcare query into ONE of these known specialties:
{specialty_list}

If it names a specialty not exactly listed, map it to the closest matching category above.

EXAMPLES:
Query: "find me an audiologist for hearing loss"
Output: {{"specialty": "ENT (Ear, Nose, Throat)"}}
Query: "I need a rheumatologist for joint pain"
Output: {{"specialty": "Orthopedics"}}
Query: "looking for an oncologist"
Output: {{"specialty": "Internal Medicine"}}

Respond with ONLY valid JSON: {{"specialty": "<exact name from the list above, or null if truly nothing fits>"}}
No explanation, no markdown, just the JSON.

Query: {query}
"""


async def _infer_specialisation_llm(query: str) -> Optional[str]:
    """LLM fallback for when infer_specialisation()'s keyword registry finds
    no match -- covers specialty names/phrasings the registry can't enumerate
    (e.g. "audiologist", "rheumatologist"), mapping to the closest known
    SPECIALTY_REGISTRY entry so slug/german-term lookup still works downstream.
    Only called from find_doctors(); never from the cheap routing-level checks."""
    try:
        import json
        import re as _re
        from backend.llm_factory import get_llm
        from langchain_core.messages import SystemMessage

        specialty_list = "\n".join(f"- {name}" for name in SPECIALTY_REGISTRY)
        llm = get_llm(temperature=0.0)
        resp = await llm.ainvoke([
            SystemMessage(content=SPECIALTY_CLASSIFIER_PROMPT.format(
                specialty_list=specialty_list, query=query,
            )),
        ])
        raw = resp.content.strip()
        json_match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not json_match:
            return None
        parsed = json.loads(json_match.group())
        specialty = parsed.get("specialty")
        return specialty if specialty in SPECIALTY_REGISTRY else None
    except Exception as e:
        logger.warning(f"Specialty LLM fallback failed: {e}")
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
        german_term = SPECIALTY_REGISTRY.get(specialization, {}).get("german_search_term") or specialization or query
        search_query = f"{german_term} Arzt {city}"
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
                '- "phone": (string, phone number if found, otherwise "-")\n'
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

                # Filter out generic website and article titles from being parsed as doctors
                is_real_doctor = any(p in doc_name for p in ["Dr.", "med.", "Prof.", "Herr", "Frau", "Praxis", "Klinik", "Therapie", "Center", "Zentrum", "Gemeinschaftspraxis"]) or any(k in doc_name.lower() for k in ["arzt", "therapeut", "psycholog", "neurolog", "kardiolog", "zahnarzt", "psychiatr", "hospital", "apotheke"])
                has_blog_keywords = any(bk in doc_name.lower() for bk in ["guide", "how to", "insurance", "feather", "expat", "mind", "handbook", "germany", "cover", "health", "system", "counseling"])
                if not is_real_doctor or has_blog_keywords:
                    logger.info(f"Skipping generic web search result title as doctor: {doc_name}")
                    continue

                # Extract address
                addr_match = addr_pattern.search(snippet)
                if not addr_match:
                    addr_match = addr_pattern.search(title)
                address = addr_match.group(1).strip() if addr_match else f"{city}, Germany"

                # Extract phone
                phone_match = phone_pattern.search(snippet)
                phone = phone_match.group(1).strip() if phone_match else "-"

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
    Find doctors matching the query by merging direct scraping and Tavily results.

    Returns
    -------
    {doctors: [...], hospitals: [...], inferred_specialisation: str|None, city: str}
    """
    inferred = specialization or infer_specialisation(query)
    if not inferred:
        # Keyword registry found nothing -- ask the LLM to map novel specialty
        # phrasings (e.g. "audiologist", "rheumatologist") to the closest known
        # SPECIALTY_REGISTRY entry, instead of silently giving up. This is a
        # deliberately narrow fallback: only invoked here (not from the cheap
        # routing-level checks in supervisor_agent.py / detect_and_search_doctors_inline),
        # since find_doctors() is already a multi-step network operation where
        # one more classification call is a proportionally small cost.
        inferred = await _infer_specialisation_llm(query)

    entry = SPECIALTY_REGISTRY.get(inferred, {})
    arzt_auskunft_slug = entry.get("arzt_auskunft_slug")
    # German terms to filter by when the arzt-auskunft.de page we load isn't
    # already specialty-specific (e.g. no dedicated URL slug for this
    # specialty) -- lets the scraper discard wrong-specialty doctors instead
    # of silently returning whatever's on the unfiltered general city page.
    filter_keywords = entry.get("german_search_term", "").split() or None

    # 1. Run direct scraper
    scraped_doctors = []
    try:
        logger.info("Running direct scraper for doctors")
        scraped_doctors = await scrape_arzt_auskunft(
            query=query,
            arzt_auskunft_slug=arzt_auskunft_slug,
            city=city,
            limit=limit,
            filter_keywords=filter_keywords,
        )
    except Exception as e:
        logger.error(f"Direct scraper error: {e}")

    # 2. Run Tavily web search
    logger.info("Searching for doctors via Tavily web search")
    tavily_doctors = await _tavily_doctor_search(query, inferred, city, limit)

    # 3. Merge results (deduplicate by name)
    seen_names = set()
    merged_doctors = []
    
    for doc in scraped_doctors:
        name_clean = doc["name"].lower().strip()
        if name_clean not in seen_names:
            seen_names.add(name_clean)
            merged_doctors.append(doc)

    for doc in tavily_doctors:
        name_clean = doc["name"].lower().strip()
        if name_clean not in seen_names:
            # Check for substring matches to prevent duplicates (e.g. Dr. Müller vs. Müller)
            if not any(seen in name_clean or name_clean in seen for seen in seen_names):
                seen_names.add(name_clean)
                merged_doctors.append(doc)

    # Enrich with map entries formatting
    doctors = [format_doctor_map_entry(d) for d in merged_doctors[:limit]]
    hospitals = [format_doctor_map_entry(h) for h in FALLBACK_HOSPITALS[:3]]

    return {
        "doctors": doctors,
        "hospitals": hospitals,
        "inferred_specialisation": inferred,
        "city": city,
        "source": "arzt-auskunft.de / jameda.de / kvno.de",
        "source_url": "https://www.arzt-auskunft.de",
    }


async def detect_and_search_doctors_inline(user_input: str, lang: str) -> Dict[str, Any] | None:
    """
    Detect doctor search intent and perform the search inline, returning results to be injected into agent contexts.
    """
    t = user_input.lower()
    doctor_keywords = [
        "doctor", "therapist", "physician", "gp", "psychotherapist", "psychiatrist",
        "arzt", "therapeut", "psychotherapeut", "psychiater", "hausarzt", "klinik", "hospital",
        "doktor", "hekim", "psikolog", "psikoterapist", "лікар", "терапевт", "психотерапевт"
    ]
    search_keywords = [
        "find", "search", "recommend", "look for", "suchen", "finden", "empfehlen", "bul", "ara", "знайти", "шукати"
    ]
    # Also trigger on any recognised medical specialty name (e.g. "gynecologist",
    # "cardiologist") even without a generic word like "doctor"/"arzt" present —
    # reuses infer_specialisation()'s specialty map as the single source of truth.
    has_doctor = any(dk in t for dk in doctor_keywords) or infer_specialisation(user_input) is not None
    has_search = any(sk in t for sk in search_keywords)
    has_direct_city = "oberhausen" in t and has_doctor
    
    if has_direct_city or (has_doctor and has_search):
        logger.info(f"Inline doctor search triggered for: '{user_input}'")
        try:
            return await find_doctors(query=user_input, language=lang)
        except Exception as e:
            logger.error(f"Error calling find_doctors inline: {e}")
    return None
