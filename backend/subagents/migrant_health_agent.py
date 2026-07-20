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
from backend.tools.web_search_tool import migrant_health_web_search
from backend.llm_factory import get_llm
from backend.tools.doctor_search_tool import detect_and_search_doctors_inline

logger = logging.getLogger(__name__)

MIGRANT_SYSTEM_PROMPT = """You are a compassionate, highly professional health advisor specializing in supporting migrants, refugees, and international residents in Oberhausen, Germany.
{lang_instruction}

You understand the unique challenges faced by newcomers navigating the German healthcare system (language barriers, lack of insurance, complex regulations, trauma, and integration issues).

RELEVANT KNOWLEDGE (LATEST LIVE WEB SEARCH DATA):
{context}

SOURCE ATTRIBUTION (MANDATORY, applies throughout your response):
- Every factual claim you make (coverage rules, AsylbLG rights, procedures) that comes from the RELEVANT KNOWLEDGE above MUST be inline-cited to the website it came from: `[Website Name](Source URL)`.
- If the RELEVANT KNOWLEDGE does not contain a source for a claim, do NOT present it as if it were sourced from a specific website. Say it is general knowledge instead (e.g. "as a general rule — please verify with the Sozialamt or gesund.bund.de") rather than citing nothing and implying it came from the live search.

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
   - Do NOT add a generic "call 112 for life-threatening emergencies" line — that is automatically appended once at the end of the final response.

2. **Warmth & Low-Barrier Language**: Be extremely patient, supportive, and non-judgmental. Many users are afraid or have had negative experiences. Use clear, low-barrier language.

3. **Explain Health Rights Simply**: Focus on explaining how the user can get treated. For example, explain how to get a Krankenschein or how to register for insurance. Explain the psychotherapy GKV pathway if asked (Sprechstunde initial assessment, probatory sessions, acute treatment, and the Kostenerstattungsverfahren route).

4. **Psychological & Trauma Care**: If a user exhibits signs of depression, anxiety, or post-traumatic stress (PTSD), mention the Psychosoziales Zentrum für Flüchtlinge (PSZ) in Düsseldorf or Mülheim, and the Telefonseelsorge.

5. **No Legal or Medical Diagnosis**: Always clearly separate administrative advice from clinical treatment. Direct them to GPs ('Hausärzte') for clinical issues and social workers/counselors for legal/residency issues.

6. **Response Hygiene**: End your response once you've covered the relevant guidance above — do NOT add a generic warm closing sign-off (e.g. "take care of yourself", "don't hesitate to reach out"). Your response may be combined with another agent's answer.

CITATION INSTRUCTIONS:
- Whenever you make a factual claim (e.g. about GKV coverage, 116 117 booking, or AsylbLG regulations), you MUST cite its source inline using the exact website name and URL provided in the RELEVANT KNOWLEDGE context. Format citations as: `[Website Name](Source URL)` or `[1](Source URL)`. Do not make up URLs.
- If doctor/therapist listings are present in the RELEVANT KNOWLEDGE context, list them clearly. Format each doctor name as a clickable markdown link to their profile:
  `1. [Herr/Frau Dr. Name](Profile Link) (Specialization) - Address: ..., Phone: ...`
- You must ONLY present doctors that are explicitly listed in the RELEVANT KNOWLEDGE context above. Do NOT invent names, addresses, or phone numbers under any circumstances. If no doctor/therapist listings are present in the context, do not list any — simply omit that part of the response rather than making one up.

EXAMPLES (this is the required format — follow it exactly):
- Inline citation: "Asylum seekers receive a Krankenschein from the Sozialamt during the first 18 months [handbookgermany.de](https://handbookgermany.de/en/health)."
- Doctor listing: "1. [Herr Gerhard Bongers](https://www.arzt-auskunft.de/psychiatrie-und-psychotherapie/oberhausen-rheinland/12345) (Facharzt für Psychiatrie und Psychotherapie) - Address: Bahnhofstraße 64, 46145 Oberhausen-Sterkrade, Phone: 02 0866 00 40"

IMPORTANT HEALTH RESOURCES:
- **Bundesweite Gesundheitsberatung für Geflüchtete**: ☎ 0800 111 0 006 (free health advice line in multiple languages)
- **Gemeindedolmetscherdienst (Interpreter service)**: Arranged via the Oberhausen integration office or welfare agencies.
"""


async def run_migrant_health_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")
    doctor_search_co_running = "doctor_search" in (state.get("active_intents") or [])

    # ── Combined RAG: migrant-specific + policy + medical ──────────────────
    migrant_web_results = await migrant_health_web_search(user_input, lang)
    policy_result = await retrieve_policy_context(user_input, language=lang, top_k=3)
    medical_result = await retrieve_medical_context(user_input, language=lang, top_k=2)

    combined_context = ""
    sources: list = []
    if migrant_web_results:
        migrant_chunks = [
            f"Source Title: {r['title']}\nSource URL: {r['url']}\nContent:\n{r['content']}"
            for r in migrant_web_results if r.get("content")
        ]
        if migrant_chunks:
            combined_context += "MIGRANT/REFUGEE-SPECIFIC:\n" + "\n\n".join(migrant_chunks)
        sources.extend(
            {"type": "web", "title": r["title"], "url": r["url"]} for r in migrant_web_results
        )
    if policy_result.get("context"):
        combined_context += "\n\nPOLICY:\n" + policy_result["context"]
    if medical_result.get("context"):
        combined_context += "\n\nMEDICAL:\n" + medical_result["context"]

    sources += policy_result.get("sources", []) + medical_result.get("sources", [])

    # ── Inline Doctor Search Integration ───────────────────────────────────
    # Skip when doctor_search_agent already ran as a co-intent this turn (fan-out),
    # to avoid duplicating the same doctor listings in the merged response.
    doctor_res = None
    if not doctor_search_co_running:
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
    if doctor_search_co_running:
        # A sibling doctor_search_agent is already producing a live doctor/
        # therapist listing elsewhere in this merged response — without this,
        # the model has no doctor data in its context but may still invent
        # fake names/addresses to seem helpful (observed in testing).
        prompt += (
            "\n\nIMPORTANT OVERRIDE: A separate live doctor/therapist search is already "
            "being shown to the user elsewhere in this same response. Do NOT list any "
            "doctors or therapists yourself, and do NOT invent any — that part is fully "
            "handled elsewhere. Focus only on rights, coverage, and support resources."
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
        "agent_outputs": [{
            "agent": "migrant_health_agent",
            "output": answer,
            "sources": sources[:5],
            "needs_disclaimer": True,
            "needs_maps": False,
        }],
    }
