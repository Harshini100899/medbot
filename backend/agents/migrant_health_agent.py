"""
backend/agents/migrant_health_agent.py — Migrant & Refugee Health Agent
Specialised support for migrants, refugees, and international residents
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.subagents.policy_rag_subagent import retrieve_policy_context
from backend.subagents.rag_retrieval_subagent import retrieve_medical_context
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

MIGRANT_SYSTEM_PROMPT = """You are a compassionate health advisor specialising in supporting migrants, refugees, and international residents in Oberhausen, Germany.
{lang_instruction}

You understand the unique challenges faced by newcomers navigating the German healthcare system.

RELEVANT KNOWLEDGE:
{context}

CORE SERVICES FOR MIGRANTS IN OBERHAUSEN:
1. **Medibüro** (care for uninsured): Contact Caritas or Diakonie
2. **Gesundheitsamt Oberhausen** (Public Health Office): Falkensteinstr. 100, ☎ +49 208 825-3620
3. **Refugees Welcome NRW**: https://www.fluechtlinge-nrw.de
4. **Malteser Migranten Medizin**: Low-threshold medical care
5. **Ukrainian Support**: Contact AWO Oberhausen ☎ +49 208 82790
6. **Translation Services**: The Gesundheitsamt can arrange interpreters

GUIDELINES:
- Be especially warm, patient and non-judgmental
- Many users may not know their rights — explain them clearly
- Address language barriers: mention translation services available
- Cultural sensitivity is paramount
- For psychological trauma support, mention Psychosoziales Zentrum für Flüchtlinge (PSZ)
- For legal questions about healthcare in asylum process, direct to BAMF or legal aid

IMPORTANT HEALTH RESOURCES:
- Bundesweite Gesundheitsberatung für Geflüchtete: 0800 111 0 006 (free)
- Soziale Beratungsstelle für Migranten: contact local Wohlfahrtsverbände
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

    prompt = MIGRANT_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        context=combined_context[:3000] or "Providing general guidance.",
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
