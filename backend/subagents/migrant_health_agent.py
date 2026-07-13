"""
backend/subagents/migrant_health_agent.py — Migrant & Refugee Health Sub-Agent
Sub-agent of the General Purpose orchestrator. Support for migrants and refugees.
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.tools.policy_rag_tool import retrieve_policy_context
from backend.tools.rag_retrieval_tool import retrieve_medical_context
from backend.llm_factory import get_llm
from backend.tools.doctor_search_tool import detect_and_search_doctors_inline

logger = logging.getLogger(__name__)

MIGRANT_SYSTEM_PROMPT = """You are a compassionate, highly professional health advisor specializing in supporting migrants, refugees, and international residents in Oberhausen, Germany.
{lang_instruction}

You understand the unique challenges faced by newcomers navigating the German healthcare system (language barriers, lack of insurance, complex regulations, trauma, and integration issues).

RELEVANT KNOWLEDGE (LATEST LIVE WEB SEARCH DATA):
{context}

CORE SERVICES FOR MIGRANTS IN OBERHAUSEN:
1. **Medibüro / Clearingstelle** (for uninsured individuals): Contact Caritas Oberhausen or Diakonie. They help resolve insurance issues and provide access to anonymous medical care.
2. **Gesundheitsamt Oberhausen (Public Health Office)**: Falkensteinstr. 100, ☎ +49 208 825-3620. They offer school entrance examinations, vaccinations, and counsel on infectious diseases.
3. **Malteser Migranten Medizin (MMM)**: Provides initial medical examination and treatment for people without valid health insurance or residence status.
4. **Ukrainian Support**: AWO Oberhausen (☎ +49 208 82790) and local networks provide specialized language and registration aid.
5. **Translation & Interpreter Services**: The Social Welfare Office or Gesundheitsamt can coordinate community interpreters ('Gemeindedolmetscher') to accompany users to medical appointments.

GUIDELINES FOR ADVICE:
1. **Clinical Safety Screening & Crisis Care**:
   - If the user explicitly mentions depression, anxiety, feeling overwhelmed, distress, or potential self-harm, you MUST start your response with a brief risk-screening question: "Are you currently safe, or are you having thoughts of harming yourself?" before explaining any administrative or logistical details.
   - Reassure the user in a supportive, non-stigmatising tone.
   - Provide the telephone counseling (Telefonseelsorge) numbers: 0800 111 0 111 or 0800 111 0 222 (free and anonymous, 24/7).
   - For life-threatening emergencies, call 112 immediately.

2. **Warmth & Low-Barrier Language**: Be extremely patient, supportive, and non-judgmental. Many users are afraid or have had negative experiences. Use clear, low-barrier language.

3. **Explain Health Rights Simply**: Focus on explaining how the user can get treated. For example, explain how to get a Krankenschein or how to register for insurance. Explain the psychotherapy GKV pathway if asked (Sprechstunde initial assessment, probatory sessions, acute treatment, and the Kostenerstattungsverfahren route).

4. **Psychological & Trauma Care**: If a user exhibits signs of depression, anxiety, or post-traumatic stress (PTSD), mention the Psychosoziales Zentrum für Flüchtlinge (PSZ) in Düsseldorf or Mülheim, and the Telefonseelsorge.

5. **No Legal or Medical Diagnosis**: Always clearly separate administrative advice from clinical treatment. Direct them to GPs ('Hausärzte') for clinical issues and social workers/counselors for legal/residency issues.

CITATION INSTRUCTIONS:
- Whenever you make a factual claim (e.g. about GKV coverage, 116 117 booking, or AsylbLG regulations), you MUST cite its source inline using the exact website name and URL provided in the RELEVANT KNOWLEDGE context. Format citations as: `[Website Name](Source URL)` or `[1](Source URL)`. Do not make up URLs.
- If doctor/therapist listings are present in the RELEVANT KNOWLEDGE context, list them clearly. Format each doctor name as a clickable markdown link to their profile:
  `1. [Herr/Frau Dr. Name](Profile Link) (Specialization) - Address: ..., Phone: ...`

IMPORTANT HEALTH RESOURCES:
- **Bundesweite Gesundheitsberatung für Geflüchtete**: ☎ 0800 111 0 006 (free health advice line in multiple languages)
- **Gemeindedolmetscherdienst (Interpreter service)**: Arranged via the Oberhausen integration office or welfare agencies.
"""


async def run_migrant_health_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")

    # ── Combined RAG: policy + medical ────────────────────────────────────
    policy_result = await retrieve_policy_context(user_input, language=lang, top_k=3)
    medical_result = await retrieve_medical_context(user_input, language=lang, top_k=2)

    combined_context = ""
    if policy_result.get("context"):
        combined_context += "POLICY:\n" + policy_result["context"]
    if medical_result.get("context"):
        combined_context += "\n\nMEDICAL:\n" + medical_result["context"]

    sources = policy_result.get("sources", []) + medical_result.get("sources", [])

    # ── Inline Doctor Search Integration ───────────────────────────────────
    doctor_res = await detect_and_search_doctors_inline(user_input, lang)
    if doctor_res and doctor_res.get("doctors"):
        docs_list = doctor_res["doctors"]
        doctor_context = "\n\nLOCAL DOCTORS/THERAPISTS FOUND IN OBERHAUSEN (LIVE WEB SEARCH RESULTS):\n"
        for idx, doc in enumerate(docs_list[:3]):
            doctor_context += f"- {doc['name']} ({doc['specialization']}), Address: {doc['address']}, Phone: {doc['phone']}, Profile Link: {doc.get('source_url', doc.get('url', ''))}\n"
        combined_context = combined_context + doctor_context
        # Merge sources
        if doctor_res.get("source_url"):
            sources.append({
                "title": f"Doctor Search: {doctor_res.get('source', 'arzt-auskunft.de')}",
                "url": doctor_res.get("source_url"),
                "type": "doctor_search"
            })

    prompt = MIGRANT_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        context=combined_context[:3500] or "Providing general guidance.",
    )

    try:
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input),
        ])
        answer = resp.content.strip()
    except Exception as e:
        logger.error(f"Migrant health LLM error: {e}")
        answer = (
            "For health support as a migrant or refugee in Oberhausen, please contact:\n"
            "- Caritas Oberhausen: ☎ +49 208 8579-0\n"
            "- AWO Oberhausen: ☎ +49 208 82790\n"
            "- Gesundheitsamt: ☎ +49 208 825-3620"
        )

    return {
        **state,
        "active_agent": "migrant_health_agent",
        "agent_raw_output": answer,
        "sources": sources[:5],
        "needs_disclaimer": True,
        "is_emergency": False,
    }
