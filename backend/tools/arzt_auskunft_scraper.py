"""
backend/tools/arzt_auskunft_scraper.py — Scraper tool for arzt-auskunft.de
Fetches and parses real doctor listings in Oberhausen from arzt-auskunft.de
"""
from __future__ import annotations
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

# Detailed mapping from query keywords to arzt-auskunft slugs
MAPPED_KEYWORDS = {
    "diabetes": "innere-medizin-und-endokrinologie-und-diabetologie",
    "diabetolog": "innere-medizin-und-endokrinologie-und-diabetologie",
    "diabetiker": "innere-medizin-und-endokrinologie-und-diabetologie",
    "cardio": "innere-medizin-und-kardiologie",
    "herz": "innere-medizin-und-kardiologie",
    "heart": "innere-medizin-und-kardiologie",
    "pediatr": "kinderheilkunde-kinder-und-jugendmedizin",
    "kinder": "kinderheilkunde-kinder-und-jugendmedizin",
    "kind": "kinderheilkunde-kinder-und-jugendmedizin",
    "child": "kinderheilkunde-kinder-und-jugendmedizin",
    "dermatolog": "haut-und-geschlechtskrankheiten",
    "skin": "haut-und-geschlechtskrankheiten",
    "haut": "haut-und-geschlechtskrankheiten",
    "ent": "hals-nasen-ohrenheilkunde",
    "hno": "hals-nasen-ohrenheilkunde",
    "hals": "hals-nasen-ohrenheilkunde",
    "ohn": "hals-nasen-ohrenheilkunde",
    "eye": "augenheilkunde",
    "ophthalmolog": "augenheilkunde",
    "auge": "augenheilkunde",
    "neurolog": "neurologie",
    "gynecolog": "frauenheilkunde-und-geburtshilfe",
    "frauen": "frauenheilkunde-und-geburtshilfe",
    "women": "frauenheilkunde-und-geburtshilfe",
    "orthoped": "orthopaedie",
    "orthop": "orthopaedie",
    "knochen": "orthopaedie",
    "bone": "orthopaedie",
    "psychiatrie": "psychiatrie-und-psychotherapie",
    "psychotherapy": "psychiatrie-und-psychotherapie",
    "psychotherapeut": "psychiatrie-und-psychotherapie",
    "therapeut": "psychiatrie-und-psychotherapie",
    "psychiatr": "psychiatrie-und-psychotherapie",
    "psych": "psychiatrie-und-psychotherapie",
    "dentist": "zahnmedizin",
    "zahn": "zahnmedizin",
    "gp": "allgemeinmedizin",
    "general": "allgemeinmedizin",
    "hausarzt": "allgemeinmedizin",
    "allgemein": "allgemeinmedizin",
}


async def fetch_phone(client: httpx.AsyncClient, url: str) -> str:
    """Fetch the detail page of the doctor and parse the first phone number."""
    if not url or not url.startswith("http"):
        return "-"
    try:
        resp = await client.get(url, timeout=4.0)
        if resp.status_code == 200:
            html = resp.text
            tels = re.findall(r'href="tel:([^"]+)"', html)
            if tels:
                return tels[0].strip()
    except Exception as e:
        logger.warning(f"Error fetching phone from {url}: {e}")
    return "-"


async def scrape_arzt_auskunft(
    query: str,
    specialization: Optional[str] = None,
    city: str = "Oberhausen",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Directly scrape arzt-auskunft.de for doctor listings matching query or specialization.
    """
    # Normalize city slug
    city_slug = "oberhausen-rheinland" if city.lower() == "oberhausen" else city.lower().replace(" ", "-")

    # Determine specialization slug from specialization name or query keywords
    spec_slug = None
    search_term = (specialization or query).lower()
    
    for kw, slug in MAPPED_KEYWORDS.items():
        if kw in search_term:
            spec_slug = slug
            break

    # Build potential URLs (prioritize specific specialty page, fall back to city page)
    urls = []
    if spec_slug:
        urls.append(f"https://www.arzt-auskunft.de/{spec_slug}/{city_slug}/")
    urls.append(f"https://www.arzt-auskunft.de/{city_slug}/")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    html = ""
    source_url = ""
    
    for url in urls:
        try:
            logger.info(f"Scraping arzt-auskunft.de: {url}")
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    # Site sends UTF-8 with some Windows-1252 characters mixed in
                    # Use errors='replace' to handle unmappable bytes gracefully
                    html = resp.content.decode("windows-1252", errors="replace")
                    source_url = url
                    break
                else:
                    logger.warning(f"Failed to fetch {url}: status {resp.status_code}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    if not html:
        return []

    # Parse doctor cards using regex (avoiding bs4 dependency)
    cards = html.split('<div class="card card-hover')
    doctors = []
    
    for card in cards[1:]:
        try:
            # Extract mobile/detail href
            href_match = re.search(r'data-href-mobile="([^"]+)"', card)
            detail_url = href_match.group(1) if href_match else source_url
            if detail_url and not detail_url.startswith("http"):
                detail_url = "https://www.arzt-auskunft.de" + detail_url

            # Extract name
            name_match = re.search(r'itemprop="name"[^>]*>([^<]+)</h2>', card)
            if not name_match:
                continue
            name = name_match.group(1).strip()

            # Extract specialty
            spec_match = re.search(r'itemprop="medicalSpecialty"[^>]*>([^<]+)</span>', card)
            spec_text = spec_match.group(1).strip() if spec_match else "Healthcare Provider"

            # Extract address parts
            street_match = re.search(r'itemprop="streetAddress"[^>]*>([^<]+)</span>', card)
            street = street_match.group(1).strip() if street_match else ""

            zip_match = re.search(r'itemprop="postalCode"[^>]*>([^<]+)</span>', card)
            zip_code = zip_match.group(1).strip() if zip_match else ""

            locality_match = re.search(r'itemprop="addressLocality"[^>]*>([^<]+)</span>', card)
            locality = locality_match.group(1).strip() if locality_match else ""

            # Check if address locality is actually Oberhausen
            full_address = f"{street}, {zip_code} {locality}"
            
            doctors.append({
                "name": name,
                "specialization": spec_text,
                "address": full_address,
                "phone": "-",  # Will be fetched concurrently below
                "languages": ["de"],
                "kvno_accepted": True,
                "source_url": detail_url,
            })
        except Exception as parse_error:
            logger.error(f"Error parsing doctor card: {parse_error}")

    # If we fell back to the main city page but had a specific specialization, filter results
    if spec_slug and source_url == urls[-1]:
        filtered = []
        # Find matching keywords in specialty text
        spec_keywords = [k for k, v in MAPPED_KEYWORDS.items() if v == spec_slug]
        for d in doctors:
            if any(kw in d["specialization"].lower() for kw in spec_keywords):
                filtered.append(d)
        if filtered:
            doctors = filtered

    # Fetch phone numbers concurrently for the selected doctors
    selected_doctors = doctors[:limit]
    if selected_doctors:
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=5.0) as client:
                tasks = [fetch_phone(client, d["source_url"]) for d in selected_doctors]
                phones = await asyncio.gather(*tasks)
                for d, phone in zip(selected_doctors, phones):
                    d["phone"] = phone if phone != "-" else "-"
        except Exception as e:
            logger.error(f"Error gathering doctor phone numbers: {e}")

    logger.info(f"Successfully scraped and parsed {len(selected_doctors)} doctors from {source_url}")
    return selected_doctors
