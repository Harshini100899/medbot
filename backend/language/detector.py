"""
backend/language/detector.py — Language detection using langdetect
"""
from __future__ import annotations
import logging
from typing import Tuple

from backend.config import settings

logger = logging.getLogger(__name__)

# Language name map for logging
LANG_NAMES = {
    "de": "German",
    "en": "English",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ru": "Russian",     # often confused with Ukrainian
}

# Minimum text length for reliable detection
MIN_DETECT_LENGTH = 10


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of *text*.

    Returns
    -------
    (lang_code, confidence)
        lang_code  — ISO-639-1 code restricted to SUPPORTED_LANGUAGES,
                     defaults to settings.DEFAULT_LANGUAGE if unsupported.
        confidence — 0.0–1.0
    """
    if not text or len(text.strip()) < MIN_DETECT_LENGTH:
        return settings.DEFAULT_LANGUAGE, 0.5

    try:
        from langdetect import detect_langs
        from langdetect import DetectorFactory
        DetectorFactory.seed = 42        # reproducible results

        results = detect_langs(text)
        if not results:
            return settings.DEFAULT_LANGUAGE, 0.5

        # Take the top result
        top = results[0]
        lang = top.lang.lower()
        conf = float(top.prob)

        # Ukrainian ↔ Russian ambiguity: trust slightly toward Ukrainian
        if lang == "ru" and conf < 0.75:
            lang = "uk"

        # Map to supported language or fall back
        if lang not in settings.SUPPORTED_LANGUAGES:
            lang = settings.DEFAULT_LANGUAGE
            conf = 0.5

        logger.debug(f"Detected language: {lang} ({conf:.2f}) for text: '{text[:40]}...'")
        return lang, round(conf, 3)

    except ImportError:
        logger.warning("langdetect not installed. Defaulting to English.")
        return settings.DEFAULT_LANGUAGE, 0.5
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return settings.DEFAULT_LANGUAGE, 0.5


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code.upper())


# ─── System prompts per language ──────────────────────────────────────────────
SYSTEM_PROMPT_LANG = {
    "de": (
        "Antworte auf Deutsch. Verwende klare, einfache und verständliche Sprache. "
        "Achte darauf, dass medizinische Fachbegriffe erklärt werden."
    ),
    "en": (
        "Respond in English. Translate concepts where appropriate, but keep important German administrative "
        "terms in parentheses next to their translation (e.g. 'health insurance card (Krankenkassenkarte)', "
        "'social welfare office (Sozialamt)', 'statutory health insurance (GKV)', 'Kassenpatienten', or 'Krankenschein'). "
        "This helps the user recognize them when navigating the German healthcare system."
    ),
    "tr": (
        "Türkçe yanıt ver. Alman sağlık sistemindeki önemli terimleri parantez içinde Almanca olarak belirt "
        "(örneğin: 'sosyal yardım dairesi (Sozialamt)', 'sağlık sigortası kartı (Krankenkassenkarte)', 'yasal sağlık sigortası (GKV)'). "
        "Bu, kullanıcının resmi kurumlarda bu terimleri tanımasını kolaylaştıracaktır."
    ),
    "uk": (
        "Відповідай українською мовою. Пиши зрозуміло та з повагою. Важливі німецькі адміністративні "
        "терміни вказуй у дужках німецькою мовою поруч із перекладом (наприклад: 'каса соціального захисту (Sozialamt)', "
        "'картка медичного страхування (Krankenkassenkarte)', 'державне медичне страхування (GKV)'). "
        "Це допоможе користувачеві орієнтуватися в німецькій медичній системі."
    ),
}


def get_language_instruction(lang: str) -> str:
    return SYSTEM_PROMPT_LANG.get(lang, SYSTEM_PROMPT_LANG["en"])
