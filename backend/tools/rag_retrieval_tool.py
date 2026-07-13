"""
backend/tools/rag_retrieval_tool.py — Medical Knowledge RAG Retrieval Tool
Used by the Medical Specialist agent (and the Migrant Health sub-agent).

Data sources (priority order):
  1. Tavily web search across trusted German medical sites
  2. Static fallback knowledge base
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from backend.tools.web_search_tool import medical_web_search

logger = logging.getLogger(__name__)

# ─── Static medical knowledge (always-available baseline) ────────────────────
STATIC_MEDICAL_KNOWLEDGE = [
    {
        "title": "Diabetes Management",
        "text": (
            "Diabetes mellitus is a chronic condition characterised by elevated blood glucose. "
            "Type 2 diabetes (the most common) is managed through lifestyle changes, oral medications "
            "(e.g. Metformin), and sometimes insulin injections. Regular HbA1c monitoring, foot care, "
            "eye exams, and kidney function checks are essential. In Germany, diabetologists "
            "(Fachärzte für Innere Medizin und Endokrinologie und Diabetologie) specialise in diabetes treatment. "
            "The doctor-on-call number 116 117 can assist outside surgery hours."
        ),
    },
    {
        "title": "Hypertension (High Blood Pressure)",
        "text": (
            "Hypertension (Bluthochdruck) is defined as blood pressure ≥140/90 mmHg. "
            "Treatment includes lifestyle changes (reduced salt intake, exercise, weight loss) "
            "and medications such as ACE inhibitors, beta blockers, or diuretics. "
            "Regular monitoring is essential to prevent heart attack and stroke. "
            "In Oberhausen, internists and general practitioners can diagnose and manage hypertension."
        ),
    },
    {
        "title": "Cold and Flu",
        "text": (
            "Common cold (Erkältung) is a viral upper respiratory infection. Symptoms include "
            "runny nose, sore throat, cough, and mild fever. Treatment is mainly symptomatic: "
            "rest, fluids, and over-the-counter medications. Influenza (Grippe) is more severe "
            "and may benefit from antiviral medication if caught early. Annual flu vaccination is "
            "recommended in Germany, especially for the elderly and those with chronic conditions."
        ),
    },
    {
        "title": "Fever Management",
        "text": (
            "Fever (Fieber) is generally defined as body temperature above 38°C. It is usually "
            "a sign the immune system is fighting infection. Adults can manage mild fever with "
            "paracetamol (acetaminophen) or ibuprofen, rest, and adequate fluid intake. "
            "Seek immediate care if fever exceeds 39.5°C, persists more than 3 days, or is "
            "accompanied by severe symptoms. Children with febrile seizures need emergency care (112)."
        ),
    },
    {
        "title": "COVID-19 Guidance",
        "text": (
            "COVID-19 is caused by the SARS-CoV-2 virus. Symptoms include fever, cough, "
            "difficulty breathing, loss of smell/taste. If symptomatic, isolate and contact "
            "your GP or the Corona-Hotline (0800 011 77 22 — free). Testing is available at "
            "pharmacies and GP practices. Vaccination is strongly recommended. For severe symptoms, "
            "call 112 immediately. High-risk groups should receive booster vaccinations as recommended."
        ),
    },
    {
        "title": "Mental Health Resources in Germany",
        "text": (
            "Mental health services in Germany include psychiatrists (Psychiater/Nervenheilkunde), "
            "psychotherapists (Psychotherapeuten), and counselling centres (Beratungsstellen). "
            "Emergency mental health support: Telefonseelsorge 0800 111 0 111 (free, 24/7). "
            "GKV covers most mental health treatment. Waiting times for outpatient psychotherapy "
            "can be long; urgent cases can request a fast-track (Akutbehandlung). "
            "In Oberhausen, the psychiatry department at Evangelisches Krankenhaus Oberhausen "
            "provides inpatient and outpatient services."
        ),
    },
    {
        "title": "Vaccination in Germany",
        "text": (
            "The Standing Committee on Vaccination (STIKO) publishes Germany's recommended "
            "vaccination schedule. Core vaccines include measles-mumps-rubella (MMR), "
            "tetanus-diphtheria-pertussis (TDaP), polio, varicella, hepatitis B, HPV, "
            "meningococcal, pneumococcal, and annual influenza. Vaccination is generally "
            "covered by GKV. In Oberhausen, GPs and paediatricians administer vaccinations. "
            "Travel vaccines can be obtained at travel medicine clinics."
        ),
    },
    {
        "title": "Chest Pain Assessment",
        "text": (
            "Chest pain can have many causes ranging from harmless (muscle strain, acid reflux) "
            "to life-threatening (heart attack, pulmonary embolism). Warning signs requiring "
            "immediate emergency care (112): pressure/squeezing in the chest, pain radiating "
            "to the arm or jaw, shortness of breath, sweating, and nausea. For non-emergency "
            "cardiac concerns, cardiologists (Kardiologen) in Oberhausen can provide evaluation."
        ),
    },
]


async def retrieve_medical_context(
    query: str,
    language: str = "en",
    top_k: int = 5,
    use_web_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve relevant medical documents for the query.
    1. Try Tavily web search on trusted German health domains.
    2. Static fallback disabled.
    """
    sources: List[Dict] = []
    context_chunks: List[str] = []

    # ── Web Search (Tavily → trusted medical domains) ──────────────────────────
    if use_web_fallback:
        try:
            web_results = await medical_web_search(query, language)
            for r in web_results:
                if r.get("content"):
                    chunk = f"Source Title: {r['title']}\nSource URL: {r['url']}\nContent:\n{r['content']}"
                    context_chunks.append(chunk)
                    sources.append({
                        "type": "web",
                        "title": r["title"],
                        "url": r["url"],
                        "score": r.get("score", 0.5),
                    })
            if web_results:
                logger.info(f"Web medical search returned {len(web_results)} results")
        except Exception as e:
            logger.warning(f"Medical web search failed: {e}")

    # ── Static knowledge fallback (Disabled) ───────────────────────────────────
    # Bypassed since the user requested no use of previous static data or database files

    return {
        "context": "\n\n".join(context_chunks[:top_k]),
        "sources": sources[:top_k],
        "has_context": bool(context_chunks),
    }
