"""
backend/ontology/normalizer.py — Medical Ontology Normaliser
Maps free-text medical terms to SNOMED-CT / ICD-10-GM / MeSH codes
(uses a curated local dictionary — expandable with real SNOMED API)
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── Embedded ontology snippet (SNOMED-CT / ICD-10-GM / MeSH) ────────────────
# In production, replace with full SNOMED-CT / ICD-10 database or Quickumls
ONTOLOGY: Dict[str, Dict[str, str]] = {
    # Symptoms
    "headache":         {"snomed": "25064002",  "icd10": "R51",   "mesh": "D006261"},
    "kopfschmerz":      {"snomed": "25064002",  "icd10": "R51",   "mesh": "D006261"},
    "baş ağrısı":       {"snomed": "25064002",  "icd10": "R51",   "mesh": "D006261"},
    "головний біль":    {"snomed": "25064002",  "icd10": "R51",   "mesh": "D006261"},
    "fever":            {"snomed": "386661006", "icd10": "R50.9", "mesh": "D005334"},
    "fieber":           {"snomed": "386661006", "icd10": "R50.9", "mesh": "D005334"},
    "ateş":             {"snomed": "386661006", "icd10": "R50.9", "mesh": "D005334"},
    "температура":      {"snomed": "386661006", "icd10": "R50.9", "mesh": "D005334"},
    "chest pain":       {"snomed": "29857009",  "icd10": "R07.9", "mesh": "D002637"},
    "brustschmerz":     {"snomed": "29857009",  "icd10": "R07.9", "mesh": "D002637"},
    "göğüs ağrısı":     {"snomed": "29857009",  "icd10": "R07.9", "mesh": "D002637"},
    "shortness of breath": {"snomed": "230145002", "icd10": "R06.0", "mesh": "D004417"},
    "atemnot":          {"snomed": "230145002", "icd10": "R06.0", "mesh": "D004417"},
    "nausea":           {"snomed": "422587007", "icd10": "R11.0", "mesh": "D009325"},
    "übelkeit":         {"snomed": "422587007", "icd10": "R11.0", "mesh": "D009325"},
    "bulantı":          {"snomed": "422587007", "icd10": "R11.0", "mesh": "D009325"},
    "cough":            {"snomed": "49727002",  "icd10": "R05",   "mesh": "D003371"},
    "husten":           {"snomed": "49727002",  "icd10": "R05",   "mesh": "D003371"},
    "öksürük":          {"snomed": "49727002",  "icd10": "R05",   "mesh": "D003371"},
    "back pain":        {"snomed": "161891005", "icd10": "M54.5", "mesh": "D001416"},
    "rückenschmerzen":  {"snomed": "161891005", "icd10": "M54.5", "mesh": "D001416"},
    # Conditions
    "diabetes":         {"snomed": "73211009",  "icd10": "E11",   "mesh": "D003920"},
    "hypertension":     {"snomed": "38341003",  "icd10": "I10",   "mesh": "D006973"},
    "bluthochdruck":    {"snomed": "38341003",  "icd10": "I10",   "mesh": "D006973"},
    "asthma":           {"snomed": "195967001", "icd10": "J45",   "mesh": "D001249"},
    "influenza":        {"snomed": "57386000",  "icd10": "J11",   "mesh": "D007251"},
    "grippe":           {"snomed": "57386000",  "icd10": "J11",   "mesh": "D007251"},
    "depression":       {"snomed": "35489007",  "icd10": "F32",   "mesh": "D003866"},
    "anxiety":          {"snomed": "197480006", "icd10": "F41.1", "mesh": "D001008"},
    "angst":            {"snomed": "197480006", "icd10": "F41.1", "mesh": "D001008"},
    "covid":            {"snomed": "840539006", "icd10": "U07.1", "mesh": "D000086382"},
    "covid-19":         {"snomed": "840539006", "icd10": "U07.1", "mesh": "D000086382"},
    "stroke":           {"snomed": "230690007", "icd10": "I63",   "mesh": "D020521"},
    "schlaganfall":     {"snomed": "230690007", "icd10": "I63",   "mesh": "D020521"},
    "heart attack":     {"snomed": "22298006",  "icd10": "I21",   "mesh": "D009203"},
    "herzinfarkt":      {"snomed": "22298006",  "icd10": "I21",   "mesh": "D009203"},
}

# ─── Emergency keywords (trigger emergency_agent routing) ────────────────────
EMERGENCY_TERMS = {
    "en": ["emergency", "ambulance", "heart attack", "stroke", "unconscious",
           "bleeding", "overdose", "suicide", "chest pain", "not breathing",
           "severe pain", "accident", "dying", "help", "sos"],
    "de": ["nofall", "notfall", "herzinfarkt", "schlaganfall", "bewusstlos",
           "blutung", "überdosis", "selbstmord", "brustschmerz", "ohnmacht",
           "hilfe", "unfall", "sterben", "rettung"],
    "tr": ["acil", "ambulans", "kalp krizi", "inme", "bayılma", "kanama",
           "doz aşımı", "intihar", "göğüs ağrısı", "nefes alamıyorum",
           "yardım", "kaza", "ölmek"],
    "uk": ["невідкладний", "швидка", "серцевий напад", "інсульт", "непритомний",
           "кровотеча", "передозування", "самогубство", "біль у грудях",
           "не дихає", "допомога", "аварія"],
}


def normalise(text: str, language: str = "en") -> List[Dict[str, str]]:
    """
    Find medical terms in *text* and return ontology codes.

    Returns list of dicts: [{term, snomed, icd10, mesh}]
    """
    text_lower = text.lower()
    found = []
    for term, codes in ONTOLOGY.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            found.append({"term": term, **codes})
    return found


def is_emergency(text: str, language: str = "en") -> bool:
    """Heuristic check: does the text contain emergency keywords?"""
    text_lower = text.lower()
    terms = EMERGENCY_TERMS.get(language, EMERGENCY_TERMS["en"])
    # Also check English regardless of detected language
    all_terms = set(terms) | set(EMERGENCY_TERMS["en"])
    for kw in all_terms:
        if kw in text_lower:
            return True
    # Digit patterns: "112", "110" in message also suggest emergency
    if re.search(r'\b(112|110|118|999|911)\b', text):
        return True
    return False


def get_icd10_codes(text: str) -> List[str]:
    """Quick extraction of ICD-10 codes from text."""
    pattern = r'\b[A-Z]\d{2}(?:\.\d)?\b'
    return re.findall(pattern, text)
