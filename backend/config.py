"""
backend/config.py — Centralised Settings (loaded from .env)
"""
from __future__ import annotations
import json
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "P4H MedBot - Oberhausen"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM Provider ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"       # ollama | openai | anthropic
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    # ── Databases ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL: int = 3600

    MONGODB_URL: str = "mongodb://admin:password@localhost:27017"
    MONGODB_DB: str = "medbot"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── External APIs ─────────────────────────────────────────────────────────
    TAVILY_API_KEY: Optional[str] = None
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    # ── Language ──────────────────────────────────────────────────────────────
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: List[str] = ["de", "en", "tr", "uk"]

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://localhost:8080",
    ]

    # ── RAG ───────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.4

    # ── Medical Ontology Disclaimer ───────────────────────────────────────────
    DISCLAIMER_EN: str = (
        "⚠️ This information is for general guidance only and does not replace "
        "professional medical advice. Always consult a qualified healthcare "
        "provider for medical decisions."
    )
    DISCLAIMER_DE: str = (
        "⚠️ Diese Information dient nur der allgemeinen Orientierung und ersetzt "
        "keine professionelle medizinische Beratung. Wenden Sie sich für medizinische "
        "Entscheidungen immer an einen qualifizierten Arzt."
    )
    DISCLAIMER_TR: str = (
        "⚠️ Bu bilgi yalnızca genel rehberlik amaçlıdır ve profesyonel tıbbi "
        "tavsiyenin yerini tutmaz. Tıbbi kararlar için her zaman nitelikli bir "
        "sağlık uzmanına danışın."
    )
    DISCLAIMER_UK: str = (
        "⚠️ Ця інформація призначена лише для загального керівництва та не замінює "
        "професійну медичну пораду. Завжди консультуйтеся з кваліфікованим "
        "медичним працівником для прийняття медичних рішень."
    )

    def get_disclaimer(self, lang: str) -> str:
        return {
            "de": self.DISCLAIMER_DE,
            "tr": self.DISCLAIMER_TR,
            "uk": self.DISCLAIMER_UK,
        }.get(lang, self.DISCLAIMER_EN)

    # ── Emergency Numbers ─────────────────────────────────────────────────────
    EMERGENCY_NUMBERS: dict = {
        "emergency": "112",
        "police": "110",
        "poison_control": "0800 192 40 (kostenlos)",
        "doctor_on_call": "116 117",
        "social_crisis": "0800 111 0 111",
        "youth_crisis": "0800 111 0 333",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
