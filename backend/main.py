"""
backend/main.py — FastAPI application entry point
P4H MedBot — Multilingual Medical Agent for Oberhausen
"""
from __future__ import annotations
import logging
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import settings
from backend.api.chat_router import router as chat_router
from backend.api.streaming_router import router as streaming_router
from backend.api.doctor_router import router as doctor_router
from backend.api.emergency_router import router as emergency_router

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence noisy third-party libraries regardless of DEBUG flag
for _noisy in (
    "pymongo", "pymongo.topology", "pymongo.serverSelection",
    "pymongo.connection", "pymongo.command", "pymongo.monitoring",
    "motor", "httpx", "httpcore", "watchfiles", "asyncio",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Test connections (gracefully — none are required)
    from backend.memory.redis_memory import get_redis
    from backend.db.mongodb import health_check as mongo_health

    redis = await get_redis()
    mongo_ok = await mongo_health()

    logger.info(f"  Redis:    {'✅ connected (short-term memory)' if redis else '⚠️  disabled (short-term memory off)'}")
    logger.info(f"  MongoDB:  {'✅ connected (persistence)' if mongo_ok else '⚠️  disabled (persistence off)'}")
    logger.info(f"  Langfuse: {'✅ enabled (observability)' if settings.langfuse_enabled else '⚠️  disabled (no keys)'}")
    logger.info(f"  ChromaDB: ⚠️  disabled (using live web search)")
    logger.info(f"  Doctor Data: arzt-auskunft.de (live scraper)")
    logger.info(f"  Medical Info: gesund.bund.de / gesundheitsinformation.de / Tavily")
    logger.info(f"  LLM:     {settings.LLM_PROVIDER} / {settings.OLLAMA_MODEL if settings.LLM_PROVIDER=='ollama' else ''}")

    # Pre-build the LangGraph
    try:
        from backend.graph.supervisor_graph import get_graph
        get_graph()
    except Exception as e:
        logger.error(f"Graph build failed: {e}")

    logger.info("=" * 60)
    logger.info(f"✅ MedBot ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📖 API docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down MedBot...")
    # Graceful cleanup of datastore connections and observability buffers.
    try:
        from backend.memory.redis_memory import close_redis
        from backend.db.mongodb import close_db
        from backend.observability.langfuse_tracer import flush as langfuse_flush

        await close_redis()
        await close_db()
        langfuse_flush()
    except Exception as e:
        logger.debug(f"Shutdown cleanup warning: {e}")


# ─── Application ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "P4H Multilingual Medical Chatbot for Oberhausen. "
        "Supports DE | EN | TR | UK. "
        "LangGraph multi-agent architecture with RAG, Redis, ChromaDB, and MongoDB."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(streaming_router)
app.include_router(doctor_router)
app.include_router(emergency_router)


# ─── Static files (frontend) ──────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    from backend.memory.redis_memory import get_redis
    from backend.db.mongodb import health_check as mongo_health

    redis_ok = (await get_redis()) is not None
    mongo_ok = await mongo_health()

    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": {
            "redis": "ok" if redis_ok else "disabled",
            "mongodb": "ok" if mongo_ok else "disabled",
            "chromadb": "disabled",
        },
        "data_sources": {
            "doctor_search": "arzt-auskunft.de (live scraper)",
            "medical_knowledge": "gesund.bund.de / gesundheitsinformation.de / Tavily",
            "policy_rights": "kvno.de / kvwl.de / Tavily",
        },
        "llm_provider": settings.LLM_PROVIDER,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
    }


@app.get("/info", tags=["System"])
async def app_info():
    """Application info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "architecture": {
            "level_1_supervisor": "Gateway router — emergency | medical | general",
            "level_2_specialists": ["medical_specialist", "general_purpose (orchestrator)"],
            "level_3_subagents": [
                "doctor_search_agent",
                "policy_rights_agent",
                "location_maps_agent",
                "migrant_health_agent",
            ],
            "fast_path": "emergency_agent",
        },
        "memory": {
            "short_term": "Redis (rolling history, cache, rate-limit)",
            "long_term": "MongoDB (durable conversation history)",
            "checkpointer": "LangGraph MemorySaver (in-process fallback)",
        },
        "observability": "Langfuse" if settings.langfuse_enabled else "disabled",
        "languages": settings.SUPPORTED_LANGUAGES,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
