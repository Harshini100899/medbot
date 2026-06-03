"""
backend/tools/web_search_tool.py — Web search via Tavily API
Supports targeted searches across trusted German medical/health domains.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

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
            logger.warning(f"Initial domain-restricted search failed: {search_err}")
            results = []

        # Fallback to unrestricted search if restricted returned nothing
        if not results and include_domains:
            logger.info("Restricted search returned no results. Retrying without domain restriction.")
            kwargs.pop("include_domains", None)
            try:
                response = await asyncio.wait_for(client.search(**kwargs), timeout=10.0)
                results = response.get("results", [])
            except Exception as fallback_err:
                logger.error(f"Fallback search failed: {fallback_err}")
                results = []

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
