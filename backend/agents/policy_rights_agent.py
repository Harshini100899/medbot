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

POLICY_SYSTEM_PROMPT = """You are an expert health rights advisor for migrants, refugees, and international residents in Oberhausen, Germany.
{lang_instruction}

Use the following policy knowledge to answer the user's question:

POLICY KNOWLEDGE:
{context}

GUIDELINES:
- Explain health insurance rights clearly (GKV, AsylbLG, EHIC)
- Mention relevant German laws and regulations by name when applicable
- Be sensitive to the user's potential situation as a migrant or refugee
- Always recommend consulting the local Sozialamt or a social worker for complex cases
- Provide practical steps (what forms to fill, where to go, what to say)
- Mention relevant organisations in Oberhausen: Caritas, Diakonie, AWO, KFD
- Note that legal situations can be complex and a professional consultation is always recommended

KEY CONTACTS IN OBERHAUSEN:
- Sozialamt Oberhausen: Schwartzstr. 72, 46045 Oberhausen, ☎ +49 208 825-0
- Ausländerbehörde (Immigration Office): Schwartzstr. 72, ☎ +49 208 825-3333
- Caritas Oberhausen: Martinstr. 26, ☎ +49 208 8579-0
- AWO Oberhausen: Marktstr. 69, ☎ +49 208 82790
- BAMF Außenstelle: Wirmerstraße 17, 46049 Oberhausen
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
