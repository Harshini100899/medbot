"""
backend/response_builder/builder.py — Response Builder
Assembles final, formatted, multilingual response with disclaimers, sources, maps
"""
from __future__ import annotations
import logging
import re
import time
from typing import Dict, Any, List

from backend.graph.state import MedBotState
from backend.config import settings

logger = logging.getLogger(__name__)

# Subagent prompts instruct against adding a generic "call 112" reminder (this
# builder always appends one definitive version below), but LLMs don't always
# follow negative instructions -- especially when 2+ agents each independently
# tend to add their own safety reminder. Strip any duplicate before appending
# the one guaranteed copy, so it doesn't show up 2-3x in a fanned-out response.
# Requires an actual call-to-action near "112" (not just co-occurring with an
# emergency-related word), so e.g. "visits cost around 112 euros" is untouched.
_CALL_112_RE = re.compile(
    r"\b(call|dial|anrufen|wählen)\b[^.!?\n]{0,20}\b112\b"
    r"|\b112\b[^.!?\n]{0,20}\b(anrufen|immediately|sofort|wählen)\b",
    re.IGNORECASE,
)


def _strip_duplicate_emergency_lines(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not _CALL_112_RE.search(s)]
    return " ".join(kept).strip()

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
    from langchain_core.messages import AIMessage

    lang = state.get("user_language", "en")
    is_emergency = state.get("is_emergency", False)

    agent_outputs = state.get("agent_outputs", [])
    if agent_outputs:
        # Fan-out path: merge contributions from one or more General Purpose
        # sub-agents that ran this turn.
        raw = "\n\n---\n\n".join(
            entry["output"] for entry in agent_outputs if entry.get("output")
        )
        sources = [s for entry in agent_outputs for s in entry.get("sources", [])]
        needs_disclaimer = any(entry.get("needs_disclaimer") for entry in agent_outputs)
        agent = "+".join(entry["agent"] for entry in agent_outputs) or "unknown"
    else:
        # Single-writer path: emergency_agent / medical_knowledge_agent.
        raw = state.get("agent_raw_output", "")
        sources = state.get("sources", [])
        needs_disclaimer = state.get("needs_disclaimer", False)
        agent = state.get("active_agent", "unknown")

    # Strip per-agent "call 112" reminders (unreliable prompt compliance) --
    # the definitive one is always appended below for non-emergency responses.
    if not is_emergency:
        raw = _strip_duplicate_emergency_lines(raw)

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

    # Append the AI response message to the messages list so that it is persisted by the checkpointer
    ai_msg = AIMessage(
        content=response,
        additional_kwargs={
            "agent_used": agent,
            "is_emergency": is_emergency,
        }
    )

    return {
        **state,
        "messages": [ai_msg],
        "final_response": response,
        "response_metadata": metadata,
        "active_agent": agent,
        "agent_outputs": [],  # reset the fan-out accumulator for the next turn
    }
