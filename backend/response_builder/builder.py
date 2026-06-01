"""
backend/response_builder/builder.py — Response Builder
Assembles final, formatted, multilingual response with disclaimers, sources, maps
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Any, List

from backend.graph.state import MedBotState
from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Section Headers per Language ─────────────────────────────────────────────
HEADERS = {
    "sources": {
        "en": "📚 Sources & References",
        "de": "📚 Quellen & Referenzen",
        "tr": "📚 Kaynaklar",
        "uk": "📚 Джерела та посилання",
    },
    "disclaimer": {
        "en": "⚠️ Medical Disclaimer",
        "de": "⚠️ Medizinischer Haftungsausschluss",
        "tr": "⚠️ Tıbbi Sorumluluk Reddi",
        "uk": "⚠️ Медичний відмовий від відповідальності",
    },
    "emergency_footer": {
        "en": "🚨 If this is a life-threatening emergency, call **112** immediately.",
        "de": "🚨 Bei lebensbedrohlichen Notfällen sofort **112** anrufen.",
        "tr": "🚨 Hayati tehlike durumunda hemen **112**'yi arayın.",
        "uk": "🚨 У разі загрози життю негайно телефонуйте **112**.",
    },
}


def _format_sources(sources: List[Dict], lang: str) -> str:
    if not sources:
        return ""
    header = HEADERS["sources"].get(lang, HEADERS["sources"]["en"])
    lines = [f"\n\n---\n**{header}**"]
    seen = set()
    for s in sources[:5]:
        title = s.get("title", "Unknown")
        url = s.get("url", "")
        src_type = s.get("type", "")
        if title in seen:
            continue
        seen.add(title)
        if url and url.startswith("http"):
            lines.append(f"- [{title}]({url})")
        elif title:
            lines.append(f"- {title} ({src_type})")
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_disclaimer(lang: str) -> str:
    header = HEADERS["disclaimer"].get(lang, HEADERS["disclaimer"]["en"])
    text = settings.get_disclaimer(lang)
    return f"\n\n---\n**{header}**\n{text}"


async def build_response(state: MedBotState) -> MedBotState:
    """
    Response builder node — final step in the graph.
    Assembles the complete response from agent output + metadata.
    """
    lang = state.get("user_language", "en")
    raw = state.get("agent_raw_output", "")
    sources = state.get("sources", [])
    needs_disclaimer = state.get("needs_disclaimer", False)
    is_emergency = state.get("is_emergency", False)
    agent = state.get("active_agent", "unknown")

    # Start with raw agent output
    response = raw

    # Add sources section (if not emergency — already has contacts)
    if sources and not is_emergency:
        response += _format_sources(sources, lang)

    # Add medical disclaimer
    if needs_disclaimer:
        response += _format_disclaimer(lang)

    # Add emergency footer on all non-emergency responses
    if not is_emergency:
        footer = HEADERS["emergency_footer"].get(lang, HEADERS["emergency_footer"]["en"])
        response += f"\n\n---\n{footer}"

    # Metadata for the API response
    metadata: Dict[str, Any] = {
        "agent_used": agent,
        "language": lang,
        "is_emergency": is_emergency,
        "sources_count": len(sources),
        "has_disclaimer": needs_disclaimer,
        "timestamp": int(time.time()),
    }

    return {
        **state,
        "final_response": response,
        "response_metadata": metadata,
    }
