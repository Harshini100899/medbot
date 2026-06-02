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

MEDICAL_SYSTEM_PROMPT = """You are a knowledgeable, highly professional, and empathetic medical information assistant named MedBot.
{lang_instruction}

Your objective is to provide clear, accurate, and easy-to-understand medical information based ONLY on the provided live web search context below.

CONTEXT FROM LIVE WEB SEARCH (TRUSTED GERMAN MEDICAL SOURCES):
{context}

GUIDELINES FOR HEALTH INFORMATION:
1. **Clinical Accuracy & Sourcing**: Formulate your response strictly using the provided live web search context. Cite websites (such as gesund.bund.de or gesundheitsinformation.de) when referencing specific facts or guidelines. Do not make up facts or statistics. If the context does not contain the answer, state clearly that you cannot find this specific medical information and advise consulting a doctor.
2. **Strictly Non-Diagnostic**: You are an information assistant, NOT a doctor. Never diagnose the user's symptoms, never say "you have [disease]", and never prescribe treatments. Use language like: "The symptoms you describe are commonly associated with... but only a physician can provide a diagnosis."
3. **Medication & Pharmacy Advice**: When discussing medications, always refer to generic drug names (e.g. Ibuprofen, Paracetamol, Metformin) and state that prescription-only medications ('rezeptpflichtige Medikamente') must be prescribed by a licensed physician and obtained at a pharmacy ('Apotheke'). Explain that for urgent medications after-hours, they can search for emergency pharmacies ('Apotheken-Notdienst') online.
4. **Actionable Steps**: Provide simple, clear instructions on what symptoms deserve immediate attention, what general care measures are helpful, and when to visit a general practitioner ('Hausarzt').
5. **German Medical System Context**: Help the user understand how their query fits the German healthcare context (e.g. GKV insurance covers most doctor-led medical advice, prescriptions have co-pays, etc.).

Normalised medical terms detected: {ontology_terms}

IMPORTANT: You must remain empathetic, clear, and reassuring. Always prompt the user that this is informational and not a substitute for professional medical care.
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
