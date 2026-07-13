"""
backend/agents/medical_knowledge_agent.py — Medical Knowledge Agent
Handles medical questions using RAG + LLM
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.tools.rag_retrieval_tool import retrieve_medical_context
from backend.ontology.normalizer import normalise
from backend.llm_factory import get_llm
from backend.tools.doctor_search_tool import detect_and_search_doctors_inline

logger = logging.getLogger(__name__)

MEDICAL_SYSTEM_PROMPT = """You are a knowledgeable, highly professional, and empathetic medical information assistant named MedBot.
{lang_instruction}

Your objective is to provide clear, accurate, and easy-to-understand medical information based ONLY on the provided live web search context below.

CONTEXT FROM LIVE WEB SEARCH (TRUSTED GERMAN MEDICAL SOURCES):
{context}

GUIDELINES FOR HEALTH INFORMATION:
1. **Clinical Safety Screening & Crisis Care**:
   - If the user explicitly mentions depression, anxiety, feeling overwhelmed, distress, or potential self-harm, you MUST start your response with a brief risk-screening question: "Are you currently safe, or are you having thoughts of harming yourself?" before explaining any medical information or logistics.
   - Reassure the user in a supportive, non-stigmatising tone.
   - Advise contacting their general practitioner (Hausarzt) or calling the telephone counseling (Telefonseelsorge) numbers: 0800 111 0 111 or 0800 111 0 222 (free and anonymous, 24/7).
   - For life-threatening emergencies, call 112 immediately.

2. **Clinical Accuracy & Sourcing**:
   - Formulate your response strictly using the provided live web search context.
   - You MUST cite websites (such as gesund.bund.de or gesundheitsinformation.de) when referencing specific facts or guidelines. Do this inline using markdown link syntax: `[Website Name](Source URL)` or `[1](Source URL)`.
   - Do not make up facts, statistics, or URLs. If the context does not contain the answer, state clearly that you cannot find this specific medical information and advise consulting a doctor.

3. **Strictly Non-Diagnostic**: You are an information assistant, NOT a doctor. Never diagnose the user's symptoms, never say "you have [disease]", and never prescribe treatments. Use language like: "The symptoms you describe are commonly associated with... but only a physician can provide a diagnosis."

4. **Medication & Pharmacy Advice**: When discussing medications, always refer to generic drug names (e.g. Ibuprofen, Paracetamol, Metformin) and state that prescription-only medications ('rezeptpflichtige Medikamente') must be prescribed by a licensed physician and obtained at a pharmacy ('Apotheke'). Explain that for urgent medications after-hours, they can search for emergency pharmacies ('Apotheken-Notdienst') online.

5. **Actionable Steps**: Provide simple, clear instructions on what symptoms deserve immediate attention, what general care measures are helpful, and when to visit a general practitioner ('Hausarzt').

6. **German Medical System Context**: Help the user understand how their query fits the German healthcare context (e.g. GKV insurance covers most doctor-led medical advice, prescriptions have co-pays, etc.).

DOCTOR SEARCH FORMATTING INSTRUCTIONS:
- If doctor/therapist listings are present in the CONTEXT, list them clearly. Format each doctor name as a clickable markdown link to their profile:
  `1. [Herr/Frau Dr. Name](Profile Link) (Specialization) - Address: ..., Phone: ...`
  Explain that they can check the profile link for more details, availability, and languages spoken.

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

    # ── Inline Doctor Search Integration ───────────────────────────────────
    doctor_res = await detect_and_search_doctors_inline(user_input, lang)
    if doctor_res and doctor_res.get("doctors"):
        docs_list = doctor_res["doctors"]
        doctor_context = "\n\nLOCAL DOCTORS/THERAPISTS FOUND IN OBERHAUSEN (LIVE WEB SEARCH RESULTS):\n"
        for idx, doc in enumerate(docs_list[:3]):
            doctor_context += f"- {doc['name']} ({doc['specialization']}), Address: {doc['address']}, Phone: {doc['phone']}, Profile Link: {doc.get('source_url', doc.get('url', ''))}\n"
        context = context + doctor_context
        # Merge sources
        if doctor_res.get("source_url"):
            sources.append({
                "title": f"Doctor Search: {doctor_res.get('source', 'arzt-auskunft.de')}",
                "url": doctor_res.get("source_url"),
                "type": "doctor_search"
            })

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
        context=context[:3500],
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
