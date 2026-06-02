"""
backend/agents/policy_rights_agent.py — Health Policy & Rights Agent
Handles insurance, rights, and administrative health questions
"""
from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import MedBotState
from backend.language.detector import get_language_instruction
from backend.subagents.policy_rag_subagent import retrieve_policy_context
from backend.llm_factory import get_llm

logger = logging.getLogger(__name__)

POLICY_SYSTEM_PROMPT = """You are an expert, compassionate health rights and administrative advisor for migrants, refugees, and international residents in Oberhausen, Germany.
{lang_instruction}

Your task is to answer health insurance and rights-related questions using the live web search context provided below.

POLICY KNOWLEDGE (LATEST LIVE WEB SEARCH DATA):
{context}

GUIDELINES FOR HEALTH POLICY & RIGHTS:
1. **Explain Statutory Health Insurance (GKV)**: Clearly outline who is eligible for GKV, how to register, and the concepts of co-pay ('Zuzahlung') for prescriptions and services. Explain that statutory insurance covers GP visits, hospital care, and most diagnostics.
2. **Asylum Seekers & Refugees (AsylbLG)**: Explain health rights under §4 and §6 of the Asylum Seekers Benefits Act (Asylbewerberleistungsgesetz). State that for the first 18 months in Germany, asylum seekers are entitled to treatment for acute illness and pain, requiring a health voucher ('Krankenschein') from the Social Welfare Office (Sozialamt). After 18 months, they receive an electronic health card (eGK) and standard GKV-like coverage.
3. **European Health Insurance Card (EHIC)**: Explain that citizens of EU/EEA countries can use their EHIC or GHIC for urgent, medically necessary healthcare, but should register with a GKV if staying long-term or working.
4. **Practical, Step-by-Step Directions**: Give the user exact steps: what document to obtain, where to submit it, and what organization can help them fill out forms.
5. **Oberhausen Organizations**: Point the user to local Oberhausen welfare organizations (Caritas, AWO, Diakonie) and public offices who can help them navigate health administration.

KEY CONTACTS IN OBERHAUSEN:
- **Sozialamt Oberhausen (Social Welfare Office)**: Schwartzstr. 72, 46045 Oberhausen, ☎ +49 208 825-0 (provides Krankenschein)
- **Ausländerbehörde Oberhausen (Immigration Office)**: Schwartzstr. 72, 46045 Oberhausen, ☎ +49 208 825-3333
- **Caritas Oberhausen**: Martinstr. 26, ☎ +49 208 8579-0 (offers social counseling and translation support)
- **AWO Oberhausen**: Marktstr. 69, ☎ +49 208 82790 (counseling for migrants)
- **BAMF Außenstelle**: Wirmerstraße 17, 46049 Oberhausen
"""


async def run_policy_rights_agent(state: MedBotState) -> MedBotState:
    lang = state.get("user_language", "en")
    user_input = state.get("user_input", "")

    # ── RAG retrieval ──────────────────────────────────────────────────────
    rag_result = await retrieve_policy_context(user_input, language=lang)
    context = rag_result.get("context", "")
    sources = rag_result.get("sources", [])

    prompt = POLICY_SYSTEM_PROMPT.format(
        lang_instruction=get_language_instruction(lang),
        context=context[:3000] or "No specific policy documents retrieved.",
    )

    try:
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input),
        ])
        answer = resp.content.strip()
    except Exception as e:
        logger.error(f"Policy rights LLM error: {e}")
        answer = "For health rights in Germany, please contact your local Sozialamt or a social welfare organisation like Caritas."

    return {
        **state,
        "active_agent": "policy_rights_agent",
        "agent_raw_output": answer,
        "sources": sources,
        "needs_disclaimer": False,
        "is_emergency": False,
    }
