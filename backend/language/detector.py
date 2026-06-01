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
    "de": "Antworte auf Deutsch. ",
    "en": "Respond in English. ",
    "tr": "Türkçe yanıt ver. ",
    "uk": "Відповідай українською мовою. ",
}


def get_language_instruction(lang: str) -> str:
    return SYSTEM_PROMPT_LANG.get(lang, SYSTEM_PROMPT_LANG["en"])
