"""
backend/observability/langfuse_tracer.py — Langfuse tracing (env-gated)

Wires Langfuse observability into the LangGraph pipeline as a LangChain
callback handler. Everything here is a no-op unless BOTH
``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are configured, so the app
runs identically with or without observability.

Works with both Langfuse v2 (``langfuse.callback.CallbackHandler``) and
v3 (``langfuse.langchain.CallbackHandler``).
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_import_failed = False       # set once if the langfuse package is missing
_env_exported = False        # keys pushed into os.environ for v3 auto-config


def _export_env() -> None:
    """Expose Langfuse credentials via env so the SDK auto-configures (v3)."""
    global _env_exported
    if _env_exported:
        return
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST or "")
    _env_exported = True


def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Return a LangChain ``CallbackHandler`` bound to this session, or ``None``
    when Langfuse is disabled / unavailable.

    Pass the result into ``config={"callbacks": [handler]}`` on graph runs.
    """
    global _import_failed

    if not settings.langfuse_enabled or _import_failed:
        return None

    _export_env()

    # Langfuse v2 — handler accepts credentials + session metadata directly.
    try:
        from langfuse.callback import CallbackHandler  # type: ignore

        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Langfuse v2 handler init failed (%s) — trying v3.", e)

    # Langfuse v3/v4 — handler reads credentials from env; session/user/metadata
    # are attached per-run via the LangChain runnable config, not the constructor
    # (see run_metadata() below, merged into graph.ainvoke(config=...)).
    try:
        from langfuse import Langfuse  # type: ignore
        from langfuse.langchain import CallbackHandler  # type: ignore

        # v4 requires the client to exist before a CallbackHandler can find it;
        # Langfuse() caches a singleton per public_key, so repeat calls are cheap.
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        return CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY)
    except ImportError:
        _import_failed = True
        logger.info("Langfuse not installed — observability disabled.")
        return None
    except Exception as e:
        logger.warning("Langfuse handler init failed (%s) — observability disabled.", e)
        return None


def run_metadata(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runnable-config ``metadata`` dict that binds a LangChain run to a Langfuse session/user."""
    md: Dict[str, Any] = {"langfuse_session_id": session_id, "langfuse_user_id": user_id}
    md.update(metadata or {})
    return md


def flush() -> None:
    """Flush any buffered Langfuse events (best-effort, on shutdown)."""
    if not settings.langfuse_enabled:
        return
    try:
        from langfuse import Langfuse  # type: ignore

        Langfuse().flush()
    except Exception:
        pass
