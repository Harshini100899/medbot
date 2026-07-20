"""
backend/tools/web_search_tool.py — Web search via Tavily API
Supports targeted searches across trusted German medical/health domains.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from backend.config import settings

logger = logging.getLogger(__name__)


def _matches_domain(url: str, allowed_domains: List[str]) -> bool:
    """True if url's host is one of allowed_domains or a subdomain of one."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in allowed_domains)

# ─── Trusted medical websites ─────────────────────────────────────────────────
DOCTOR_SEARCH_DOMAINS = [
    "arzt-auskunft.de",
    "arztsuche.kvno.de",
    "jameda.de",
    "kvno.de",
    "kvwl.de",
]

MEDICAL_KNOWLEDGE_DOMAINS = [
    "gesund.bund.de",
    "gesundheitsinformation.de",
    "bundesgesundheitsministerium.de",
    "kvno.de",
    "kvwl.de",
]

POLICY_DOMAINS = [
    "gesund.bund.de",
    "gesundheitsinformation.de",
    "bundesgesundheitsministerium.de",
    "kvno.de",
    "kvwl.de",
]

MIGRANT_HEALTH_DOMAINS = [
    "handbookgermany.de",
    "bamf.de",
    "diakonie.de",
    "caritas.de",
    "gesund.bund.de",
]


async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily.

    Returns list of [{title, url, content, score}]
    Falls back to empty list if Tavily key not set or unavailable.
    """
    api_key = settings.TAVILY_API_KEY
    if not api_key or api_key == "tvly-placeholder":
        logger.warning("TAVILY_API_KEY not set or is placeholder — web search disabled.")
        return []

    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=api_key)

        kwargs: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains

        import asyncio
        results = []
        try:
            response = await asyncio.wait_for(client.search(**kwargs), timeout=10.0)
            results = response.get("results", [])
        except Exception as search_err:
            logger.warning(f"Domain-restricted search failed: {search_err}")
            results = []

        # Tavily's include_domains is a bias, not a strict filter -- empirically
        # it "tops up" with out-of-domain results when too few in-domain matches
        # exist for max_results. Enforce the restriction ourselves so answers
        # can only ever be sourced from the given trusted domains.
        if include_domains:
            before = len(results)
            results = [r for r in results if _matches_domain(r.get("url", ""), include_domains)]
            if len(results) < before:
                logger.info(
                    f"Filtered {before - len(results)} out-of-domain result(s) not in "
                    f"{include_domains} for query: {query!r}"
                )

        if not results and include_domains:
            logger.info(f"No results from trusted domains {include_domains} for query: {query!r}")

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],  # truncate to save tokens
                "score": r.get("score", 0.0),
            }
            for r in results
        ]
    except ImportError:
        logger.warning("tavily-python not installed.")
        return []
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


async def doctor_web_search(query: str, city: str = "Oberhausen") -> List[Dict]:
    """Focused doctor/clinic search across trusted German directories."""
    enhanced_query = f"{query} Arzt Oberhausen"
    return await web_search(
        enhanced_query,
        max_results=5,
        search_depth="advanced",
        include_domains=DOCTOR_SEARCH_DOMAINS,
    )


async def medical_web_search(query: str, language: str = "en") -> List[Dict]:
    """Focused medical knowledge search on trusted German health domains."""
    # Enhance query for German health sites
    if language == "de":
        enhanced_query = query
    else:
        # For non-German queries, add German context to get relevant German health info
        enhanced_query = query
    return await web_search(
        enhanced_query,
        max_results=5,
        search_depth="advanced",
        include_domains=MEDICAL_KNOWLEDGE_DOMAINS,
    )


async def policy_web_search(query: str, language: str = "en") -> List[Dict]:
    """Search for health policy and rights information on German government/insurance sites."""
    return await web_search(
        query,
        max_results=4,
        search_depth="advanced",
        include_domains=POLICY_DOMAINS,
    )


async def migrant_health_web_search(query: str, language: str = "en") -> List[Dict]:
    """Search for migrant/refugee-specific health and integration guidance on trusted domains."""
    return await web_search(
        query,
        max_results=4,
        search_depth="advanced",
        include_domains=MIGRANT_HEALTH_DOMAINS,
    )
