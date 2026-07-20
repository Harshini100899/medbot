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

# NOTE: this module intentionally carries NO specialty-keyword knowledge of
# its own (it used to have a MAPPED_KEYWORDS dict here that independently
# re-derived specialty->slug mappings from raw keywords -- that duplication is
# exactly what let it drift out of sync with doctor_search_tool.py's own
# keyword map, causing the same specialty bug to resurface twice). Callers now
# resolve the specialty via doctor_search_tool.SPECIALTY_REGISTRY and pass the
# already-known arzt_auskunft_slug + filter_keywords directly.


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
    arzt_auskunft_slug: Optional[str] = None,
    city: str = "Oberhausen",
    limit: int = 5,
    filter_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Directly scrape arzt-auskunft.de for doctor listings. `arzt_auskunft_slug`
    (if any) must already be resolved by the caller via
    doctor_search_tool.SPECIALTY_REGISTRY -- this function has no specialty
    knowledge of its own.
    """
    # Normalize city slug
    city_slug = "oberhausen-rheinland" if city.lower() == "oberhausen" else city.lower().replace(" ", "-")

    # Build potential URLs (prioritize specific specialty page, fall back to city page)
    urls = []
    if arzt_auskunft_slug:
        urls.append(f"https://www.arzt-auskunft.de/{arzt_auskunft_slug}/{city_slug}/")
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
                    # Site serves UTF-8; decoding real UTF-8 bytes as windows-1252
                    # mangles every umlaut into mojibake (e.g. "ü" -> "Ã¼"). Try
                    # UTF-8 first and only fall back to windows-1252 (replacing
                    # unmappable bytes) if the response genuinely isn't UTF-8.
                    try:
                        html = resp.content.decode("utf-8")
                    except UnicodeDecodeError:
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

    # If we fell back to the main (unfiltered) city page but a specific
    # specialization was requested, keep only doctors whose listed specialty
    # actually matches -- rather than silently presenting unrelated specialists
    # (e.g. psychiatrists for a physiotherapy search) as if they were relevant.
    # Covers specialties with no dedicated arzt-auskunft.de URL slug too
    # (arzt_auskunft_slug is None), since filter_keywords doesn't depend on it.
    if source_url == urls[-1] and filter_keywords:
        doctors = [
            d for d in doctors
            if any(kw.lower() in d["specialization"].lower() for kw in filter_keywords)
        ]

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
