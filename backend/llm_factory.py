"""
backend/llm_factory.py — Unified LLM factory supporting Ollama / OpenAI / Anthropic
"""
from __future__ import annotations
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from backend.config import settings
import logging

logger = logging.getLogger(__name__)


def get_llm(streaming: bool = False, temperature: Optional[float] = None) -> BaseChatModel:
    """Return the configured chat LLM."""
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=settings.OLLAMA_MODEL,
                temperature=temp,
                base_url=settings.OLLAMA_BASE_URL,
                num_predict=settings.LLM_MAX_TOKENS,
                streaming=streaming,
            )
        except Exception as e:
            logger.error(f"Ollama init failed: {e}. Falling back to mock.")
            return _get_mock_llm()

    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temp,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=streaming,
        )

    elif provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=temp,
            api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=streaming,
        )

    elif provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.GROQ_MODEL,
            temperature=temp,
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'ollama', 'openai', 'anthropic', or 'groq'.")



def _get_mock_llm() -> BaseChatModel:
    """Fallback mock LLM for testing without an actual LLM running."""
    from langchain_core.messages import AIMessage

    class MockLLM(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "mock"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            last = messages[-1].content if messages else ""
            text = (
                f"[MOCK LLM] Received: '{last[:80]}...'\n"
                "Please configure a real LLM in .env (OLLAMA recommended for local use)."
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    logger.warning("Using MOCK LLM — no real answers. Set LLM_PROVIDER in .env.")
    return MockLLM()
