"""
backend/agents/medical_knowledge_agent.py — Medical Knowledge Agent
Handles medical questions using RAG + LLM
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.subagents.rag_retrieval_subagent import retrieve_medical_context
from backend.ontology.normalizer import normalise
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

MEDICAL_SYSTEM_PROMPT = """You are a knowledgeable, empathetic medical information assistant named MedBot.
{lang_instruction}
You have access to verified medical information. Use the following retrieved context to answer accurately.

CONTEXT FROM KNOWLEDGE BASE:
{context}

GUIDELINES:
- Provide clear, accurate medical information based on the context
- Use simple language the user can understand
- If a condition may require medical attention, say so clearly
- NEVER diagnose — you provide information, not medical diagnoses
- Mention relevant ICD-10 or SNOMED terms when appropriate for reference
- For medications: mention generic names and note that a prescription may be needed
- If you don't have sufficient information, say so honestly

Normalised medical terms detected: {ontology_terms}

IMPORTANT: Always add a medical disclaimer at the end.
"""


async def run_medical_knowledge_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # ── RAG retrieval ──────────────────────────────────────────────────────
    rag_result = await retrieve_medical_context(user_input, language=lang)
    context = rag_result.get("context", "No specific context retrieved.")
    sources = rag_result.get("sources", [])

    # ── Ontology normalisation ─────────────────────────────────────────────
    terms = normalise(user_input, lang)
    term_str = ", ".join(
        f"{t['term']} (ICD-10: {t['icd10']})" for t in terms[:3]
    ) or "none"

    # ── Build conversation history for LLM ────────────────────────────────
    history_str = ""
    if messages:
        recent = messages[-4:] if len(messages) > 4 else messages
        history_str = "\n".join(
            f"{m.type.upper()}: {m.content[:200]}" for m in recent
        )

    prompt = MEDICAL_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        context=context[:3000],
        ontology_terms=term_str,
    )

    full_prompt = prompt
    if history_str:
        full_prompt += f"\n\nCONVERSATION HISTORY:\n{history_str}"

    # ── LLM call ──────────────────────────────────────────────────────────
    try:
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=full_prompt),
            HumanMessage(content=user_input),
        ])
        answer = resp.content.strip()
    except Exception as e:
        logger.error(f"Medical knowledge LLM error: {e}")
        answer = (
            "I apologise, but I am unable to process your medical query at the moment. "
            "Please consult a healthcare professional for accurate medical advice."
        )

    return {
        **state,
        "active_agent": "medical_knowledge_agent",
        "agent_raw_output": answer,
        "sources": sources,
        "retrieved_docs": rag_result.get("context", "").split("\n\n")[:5],
        "normalised_terms": [t["term"] for t in terms],
        "needs_disclaimer": True,
        "is_emergency": False,
    }
