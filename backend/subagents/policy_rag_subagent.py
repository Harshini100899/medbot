"""
backend/subagents/policy_rag_subagent.py — Policy & Rights RAG Sub-Agent
Serves Policy/Rights Agent and Migrant Health Agent

Data sources:
  1. Tavily web search on trusted German policy/insurance domains
  2. Static policy knowledge base (always available)
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from backend.tools.web_search_tool import policy_web_search

logger = logging.getLogger(__name__)

# ─── Static policy knowledge base ────────────────────────────────────────────
STATIC_POLICY_KNOWLEDGE = {
    "en": [
        {
            "title": "Health Insurance in Germany (GKV/PKV)",
            "text": (
                "Germany has a dual health insurance system: Statutory Health Insurance (GKV – "
                "Gesetzliche Krankenversicherung) and Private Health Insurance (PKV – Private "
                "Krankenversicherung). Employees earning under the annual income threshold (~€69,300 "
                "in 2024) are typically enrolled in GKV. Contributions are shared between employer "
                "and employee. GKV covers doctor visits, hospital stays, medications, and preventive care."
            ),
        },
        {
            "title": "Health Rights for Asylum Seekers (Asylbewerberleistungsgesetz)",
            "text": (
                "Under §4 AsylbLG, asylum seekers in Germany are entitled to treatment for acute "
                "illnesses and pain. During the first 18 months, they receive a health treatment "
                "voucher (Krankenschein) from the social welfare office (Sozialamt). After 18 months, "
                "they typically receive full GKV coverage. Emergency care is always available "
                "regardless of insurance status."
            ),
        },
        {
            "title": "Health Rights for EU Citizens",
            "text": (
                "EU/EEA citizens in Germany can use their European Health Insurance Card (EHIC/GHIC) "
                "for medically necessary treatment. For longer stays, registering with GKV is advisable. "
                "In Oberhausen, the AOK Rheinland/Hamburg, Barmer, and TK are major health insurers."
            ),
        },
        {
            "title": "Rights for Uninsured Patients",
            "text": (
                "Uninsured individuals in Germany have the right to emergency care, which hospitals "
                "must provide. For non-emergency care, Medibüros (medical offices for uninsured) "
                "offer free or low-cost consultations. In NRW (North Rhine-Westphalia), contact "
                "the local Sozialamt for assistance."
            ),
        },
        {
            "title": "Prescription Medications (Rezeptpflichtige Medikamente)",
            "text": (
                "In Germany, prescription medications require a doctor's prescription (Rezept). "
                "GKV members pay a co-pay of €5–10 per prescription. Some medications are exempt. "
                "Children under 18 are exempt from co-pays. The ATC code system classifies medicines."
            ),
        },
    ],
    "de": [
        {
            "title": "Krankenversicherung in Deutschland",
            "text": (
                "Deutschland hat ein duales Krankenversicherungssystem: Gesetzliche Krankenversicherung (GKV) "
                "und Private Krankenversicherung (PKV). Arbeitnehmer unter der Jahresarbeitsentgeltgrenze "
                "(ca. 69.300 € in 2024) sind in der Regel in der GKV pflichtversichert."
            ),
        },
        {
            "title": "Gesundheitsrechte für Asylsuchende",
            "text": (
                "Nach §4 AsylbLG haben Asylsuchende Anspruch auf Behandlung akuter Erkrankungen. "
                "In den ersten 18 Monaten erhalten sie einen Krankenschein vom Sozialamt. "
                "Notfallbehandlungen sind stets verfügbar, unabhängig vom Versicherungsstatus."
            ),
        },
    ],
}


async def retrieve_policy_context(
    query: str,
    language: str = "en",
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Retrieve policy/rights information.
    1. Tavily web search on trusted German policy domains.
    2. Static fallback disabled.
    """
    context_chunks: List[str] = []
    sources: List[Dict] = []

    # ── Web search first (Tavily → trusted policy domains) ────────────────────
    try:
        web_results = await policy_web_search(query, language)
        for r in web_results:
            if r.get("content"):
                context_chunks.append(r["content"])
                sources.append({"type": "web", "title": r["title"], "url": r["url"]})
        if web_results:
            logger.info(f"Policy web search returned {len(web_results)} results")
    except Exception as e:
        logger.warning(f"Policy web search failed: {e}")

    # ── Static knowledge (Disabled) ───────────────────────────────────────────
    # Bypassed since the user requested no use of previous static data or database files

    return {
        "context": "\n\n".join(context_chunks[:top_k]),
        "sources": sources[:top_k],
        "has_context": bool(context_chunks),
    }
