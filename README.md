# P4H MedBot — Oberhausen
**Multilingual Medical AI Chatbot** · DE | EN | TR | UK

A full-stack, production-ready medical chatbot built with **LangGraph multi-agent architecture**, FastAPI, Redis, and MongoDB (with optional Langfuse observability). Designed for Oberhausen residents — especially migrants, refugees, and non-German speakers — to navigate the German healthcare system.

---

## 🏗 Architecture

```
User (DE|EN|TR|UK)
        │
        ▼
  FastAPI Gateway ─── SSE Streaming ──► Browser
        │
  Language Detection (langdetect)  +  Medical Ontology (ICD-10-GM / SNOMED / MeSH)
        │
  ┌─────▼───────────────────────────────────────────────┐
  │  LEVEL 1 — SUPERVISOR AGENT (Gateway Router)        │
  │  Cheap binary/tertiary classify: emergency | medical | general │
  └───┬─────────────────────┬───────────────────────────┘
      │ emergency           │ medical                     │ general
      ▼                     ▼                             ▼
 ┌─────────┐   ┌────────────────────────┐   ┌──────────────────────────────┐
 │Emergency│   │ LEVEL 2 — MEDICAL      │   │ LEVEL 2 — GENERAL PURPOSE    │
 │ (112)   │   │ SPECIALIST AGENT       │   │ AGENT (Orchestrator)         │
 │fast-path│   │ (clinical knowledge)   │   │ LLM sub-intent routing       │
 └────┬────┘   └───────────┬────────────┘   └──────────────┬───────────────┘
      │                    │                 ┌─────────────┼──────────────┬────────────┐
      │                    │                 ▼             ▼              ▼            ▼
      │                    │        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │                    │        │ LEVEL 3  │  │ LEVEL 3  │  │ LEVEL 3  │  │ LEVEL 3  │
      │                    │        │ Doctor   │  │ Policy & │  │ Migrant  │  │  Maps    │
      │                    │        │ Search   │  │ Rights   │  │ Health   │  │          │
      │                    │        └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
      └────────────────────┴────────────┴─────────────┴─────────────┴────────────┘
                                         │
                            ┌────────────▼───────────────┐
                            │  RESPONSE BUILDER          │
                            │  multilingual · citations  │
                            │  · disclaimer → SSE        │
                            └────────────┬───────────────┘
                                         │
   ┌─────────────────────────────────────▼─────────────────────────────────────┐
   │  MEMORY & OBSERVABILITY                                                     │
   │  Redis (short-term: rolling history, cache, rate-limit)  ·  authoritative  │
   │  MongoDB (long-term: durable conversation history)       ·  authoritative  │
   │  LangGraph MemorySaver (in-process per-turn checkpointer) ·  fallback       │
   │  Langfuse (tracing/observability, env-gated)  ·  Tavily (live web search)  │
   └────────────────────────────────────────────────────────────────────────────┘
```

> **Sub-agents use retrieval tools** in `backend/tools/` (arzt-auskunft scraper,
> Tavily web search, policy RAG, maps/geocoding) — the agents own the reasoning,
> the tools own the data access.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Languages** | German, English, Turkish, Ukrainian (auto-detected) |
| **Hierarchical agents** | 1 supervisor (gateway) + 2 Level-2 agents (Medical Specialist, General Purpose orchestrator) + 4 Level-3 sub-agents (Doctor Search, Policy & Rights, Location & Maps, Migrant Health) + Emergency fast-path |
| **RAG** | Live Tavily web search across trusted German medical/policy sites, with a static fallback knowledge base when Tavily is unavailable. *(ChromaDB code exists in the repo but is currently unused/disabled — see Database Services below.)* |
| **Short-term Memory** | Redis (session context, rate limiting, 3600s TTL) |
| **Long-term Memory** | MongoDB (conversation history, sessions) |
| **Streaming** | Server-Sent Events (SSE) — real-time token-by-token response |
| **Ontology** | SNOMED-CT, ICD-10-GM, MeSH local normalizer |
| **Emergency Fast-path** | Instant 112 banner — bypasses all LLM calls |
| **Graceful degradation** | Works without Redis/MongoDB/Tavily (reduced features) |
| **Observability** | Langfuse tracing — env-gated, no-op without keys (see Observability section) |
| **LLM Agnostic** | Ollama (local) · OpenAI · Anthropic · Groq — switch via `.env` |

---

## 📋 Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required |
| Docker + Docker Compose | Any | Optional — for Redis & MongoDB. The app runs with them disabled (graceful degradation); native installs (e.g. a local MongoDB service, Memurai/WSL for Redis) work too, no Docker required |
| Ollama | Latest | Only if `LLM_PROVIDER=ollama` (default). Not needed for `openai`/`anthropic`/`groq` |
| Git | Any | Optional |

---

## 🚀 Quick Start

### 1. Extract and enter directory
```bash
unzip Medbot.zip
cd medbot
```

### 2. Run setup (does everything)
```bash
bash setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies from `requirements.txt`
- Create `.env` from `.env.example`
- Start Redis + MongoDB via Docker Compose
- Pull the Ollama model (`llama3.2:3b` by default)
- Seed ChromaDB with medical knowledge *(legacy step — not read by the active agent pipeline, see Database Services)*

### 3. Edit `.env` (optional but recommended)
```bash
nano .env   # or any editor
```

Key settings:
```env
LLM_PROVIDER=ollama          # ollama | openai | anthropic | groq
OLLAMA_MODEL=medgemma        # Google MedGemma — medical-tuned (recommended)
TAVILY_API_KEY=tvly-...      # get free at https://tavily.com (optional)
OPENAI_API_KEY=sk-...        # only if using OpenAI instead of Ollama
GROQ_API_KEY=gsk_...         # only if using Groq instead of Ollama
LANGFUSE_PUBLIC_KEY=pk-lf-...  # optional — enables tracing, see Observability below
LANGFUSE_SECRET_KEY=sk-lf-...  # optional
```

> **Note:** Redis, MongoDB and ChromaDB are **optional**. The bot runs fully
> offline with Ollama + Tavily web search. Datastores only add short-term
> memory and conversation persistence.

### 4. Start the server
```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the app
```
http://localhost:8000
```

API docs (Swagger):
```
http://localhost:8000/docs
```

---

## 📁 Project Structure

```
medbot/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # All settings (pydantic-settings)
│   ├── llm_factory.py           # Ollama/OpenAI/Anthropic/Groq factory
│   ├── agents/                  # Level 1 + Level 2 (main agents)
│   │   ├── supervisor_agent.py       # L1 gateway router: emergency|medical|general
│   │   ├── general_purpose_agent.py  # L2 orchestrator: routes to sub-agents
│   │   ├── medical_knowledge_agent.py# L2 Medical Specialist (clinical knowledge)
│   │   └── emergency_agent.py        # 🚨 Fast-path 112 response
│   ├── subagents/               # Level 3 (leaf sub-agents under General Purpose)
│   │   ├── doctor_search_agent.py
│   │   ├── policy_rights_agent.py
│   │   ├── migrant_health_agent.py
│   │   └── location_maps_agent.py
│   ├── graph/
│   │   ├── state.py             # LangGraph state (TOP_ROUTES + GENERAL_SUBINTENTS)
│   │   └── supervisor_graph.py  # Hierarchical StateGraph + persistence + Langfuse
│   ├── memory/
│   │   ├── redis_memory.py      # Short-term memory + cache + rate limiting
│   │   └── chroma_memory.py     # Vector store ops — unused by the active pipeline (legacy)
│   ├── db/
│   │   ├── mongodb.py           # Motor async MongoDB persistence
│   │   └── seed_rag.py          # Seeds ChromaDB — unused by the active pipeline (legacy)
│   ├── observability/
│   │   └── langfuse_tracer.py   # Env-gated Langfuse callback handler
│   ├── language/
│   │   └── detector.py          # langdetect + DE/EN/TR/UK logic
│   ├── ontology/
│   │   └── normalizer.py        # SNOMED-CT / ICD-10-GM / MeSH
│   ├── response_builder/
│   │   └── builder.py           # Assembles final response + disclaimers
│   ├── tools/                   # Retrieval tools (data access for the agents)
│   │   ├── rag_retrieval_tool.py     # Medical knowledge RAG (Tavily)
│   │   ├── doctor_search_tool.py     # arzt-auskunft scraper + Tavily
│   │   ├── policy_rag_tool.py        # Policy/rights RAG
│   │   ├── maps_search_tool.py       # Places + transit lookup
│   │   ├── web_search_tool.py        # Tavily async search
│   │   ├── arzt_auskunft_scraper.py  # arzt-auskunft.de HTML scraper
│   │   └── maps_tool.py              # Google Maps + Nominatim geocoding
│   └── api/
│       ├── chat_router.py       # POST /api/chat/message
│       ├── streaming_router.py  # GET  /api/stream/chat (SSE)
│       ├── doctor_router.py     # GET  /api/doctors/search
│       └── emergency_router.py  # GET  /api/emergency/contacts
├── frontend/
│   ├── index.html               # Chat UI
│   ├── style.css                # Clean medical design
│   └── app.js                   # SSE streaming + markdown + session
├── scripts/
│   └── mongo-init.js            # MongoDB seed (doctors, pharmacies)
├── data/
│   └── chroma_db/               # ChromaDB persistent storage (auto-created)
├── docker-compose.yml           # Redis 7.4 + MongoDB 7.0
├── requirements.txt
├── .env.example
├── setup.sh
└── README.md
```

---

## 🔌 API Reference

### Chat (REST)
```http
POST /api/chat/message
Content-Type: application/json

{
  "message": "I have a headache and fever",
  "session_id": "optional-uuid"
}
```
Response:
```json
{
  "response": "...",
  "session_id": "uuid",
  "language": "en",
  "intent": "policy_rights",
  "agent": "supervisor",
  "is_emergency": false,
  "sources": [
    {"title": "Doctor Search: arzt-auskunft.de / jameda.de / kvno.de", "url": "https://www.arzt-auskunft.de", "type": "doctor_search"}
  ],
  "metadata": {}
}
```
> `intent` is one of the routing values from `TOP_ROUTES`/`GENERAL_SUBINTENTS` in `backend/graph/state.py` (e.g. `medical_knowledge`, `doctor_search`, `policy_rights`, `location_maps`, `migrant_health`, `emergency`). `agent` is currently always `"supervisor"` (hardcoded in `chat_router.py` for the UI) regardless of which node actually answered — check `intent` or the Langfuse trace to see the real node path.

### Streaming (SSE)
```http
GET /api/stream/chat?message=<text>&session_id=<id>
```
Events:
- `event: token` — `{"token": "partial text"}`
- `event: agent`  — `{"agent": "emergency"}`
- `event: done`   — `{"response": "...", "session_id": "...", "sources": [...]}`
- `event: error_event` — `{"message": "..."}`

### Doctor Search
```http
GET /api/doctors/search?query=cardiologist&language=en
GET /api/doctors/hospitals
GET /api/doctors/pharmacies
```

### Emergency Contacts
```http
GET /api/emergency/contacts
```

### Health Check
```http
GET /health
```

---

## 🧠 LLM Configuration

### Option A: Ollama (Local — Recommended for privacy)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```
Install Ollama: https://ollama.com  
Pull model: `ollama pull llama3.2:3b`

Recommended medical models:
- `llama3.2:3b` — fast, lightweight (default)
- `medllama2` — fine-tuned for medical
- `mistral:7b` — balanced quality

### Option B: OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Option C: Anthropic Claude
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### Option D: Groq (fast hosted inference)
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
```
Get a free API key at https://console.groq.com/keys. Other supported models: `openai/gpt-oss-120b`, `llama-3.1-8b-instant`.

---

## 📈 Observability (Langfuse)

Every LangGraph run (all node hops + every underlying LLM call) can be traced to [Langfuse](https://cloud.langfuse.com) for debugging and monitoring — prompts, tokens, latency, and full session replays in a web dashboard. It's **env-gated**: the app runs identically with or without it, and is a no-op until both keys below are set.

### Setup
1. Sign up free at https://cloud.langfuse.com and create a project.
2. **Settings → API Keys → Create new API key**, copy the Public and Secret keys.
3. Add to `.env`:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
   ```
4. `langfuse` is already listed in `requirements.txt`; if it's not yet installed in your venv, run `pip install -r requirements.txt` (or `pip install langfuse` directly), then restart the server:
   ```bash
   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Confirm it's active — the startup log prints:
   ```
   Langfuse: ✅ enabled (observability)
   ```

### Viewing traces
Send any chat message, then open **cloud.langfuse.com → your project → Tracing**. Each conversation turn appears as a trace named after the `session_id`, showing the full node path (e.g. `supervisor → general_purpose → policy_rights_agent → response_builder`) and every Groq/OpenAI/Anthropic/Ollama call underneath it.

> Implementation: [`backend/observability/langfuse_tracer.py`](backend/observability/langfuse_tracer.py) builds the LangChain `CallbackHandler`; [`backend/graph/supervisor_graph.py`](backend/graph/supervisor_graph.py) attaches it (plus `run_metadata()` for session/user linkage) to every `graph.ainvoke()`/`astream_events()` call.

---

## 🔍 Tavily Web Search (Optional)
Get a free API key at https://tavily.com (1000 searches/month free).
```env
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```
Without it, the bot falls back to the static in-code knowledge listed under Static Fallback Knowledge below.

---

## 🗄 Database Services

### Redis (Short-term memory)
- Session context (last 20 messages)
- Rate limiting (30 msg/min per session)
- Language cache
- TTL: 3600s

### MongoDB (Long-term persistence)
- Full conversation history
- Session metadata
- Doctor/pharmacy records

### ChromaDB (Vector store) — currently unused
`backend/memory/chroma_memory.py` and `backend/db/seed_rag.py` implement a ChromaDB vector store, but the active agent pipeline (`rag_retrieval_tool.py`, `policy_rag_tool.py`) does not call into it — `backend/main.py` reports it as disabled (`ChromaDB: ⚠️ disabled (using live web search)`), and `/health` always returns `"chromadb": "disabled"`. Medical/policy knowledge instead comes from live Tavily web search plus a static in-code fallback (`STATIC_MEDICAL_KNOWLEDGE` in `rag_retrieval_tool.py`). Treat the ChromaDB code as legacy unless you re-wire the tools to use it.

---

## 🌍 Multilingual Support

| Language | Code | Detection | Embeddings |
|---|---|---|---|
| Deutsch | `de` | ✅ | ✅ |
| English | `en` | ✅ | ✅ |
| Türkçe | `tr` | ✅ | ✅ |
| Українська | `uk` | ✅ (custom) | ✅ |

Language detection/response is handled by `backend/language/detector.py`. `EMBEDDING_MODEL` (`paraphrase-multilingual-MiniLM-L12-v2`, `backend/language/embeddings.py`) is configured but currently unused by the active pipeline — it was intended for the ChromaDB similarity search noted above, which is disabled. Actual retrieval (Tavily + static fallback) is keyword-based, not embedding-based.

---

## 🚨 Emergency Handling

Any message containing emergency keywords in DE/EN/TR/UK triggers the **Emergency Fast-path**:
- Instant response with 112 banner
- First-aid LLM guidance
- Nearest hospitals in Oberhausen
- Full emergency number table

Keywords (see `KEYWORD_RULES["emergency"]` in `backend/agents/supervisor_agent.py`): `112`, `911`, `ambulance`, `notfall`, `notruf`, `unconscious`/`ohnmächtig`, `heart attack`/`herzinfarkt`, `stroke`/`schlaganfall`, `can't breathe`/`not breathing`, `acil`, `dying`/`sterbe`, etc.

---

## 🔒 Safety & Disclaimers

All responses include a medical disclaimer in the detected language:
> ⚠️ This information is for general guidance only and does not replace professional medical advice. Always consult a qualified healthcare provider for medical decisions.

Rate limiting: 30 messages per minute per session (Redis-backed).

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check logs
python -m uvicorn backend.main:app --reload 2>&1 | head -50
```

### Redis connection error
```bash
docker-compose up -d redis
# or: redis-server --daemonize yes
```

### MongoDB connection error
```bash
docker-compose up -d mongodb
```

### Ollama model not found
```bash
ollama pull llama3.2:3b
ollama serve   # ensure running
```

### Tavily web search failing (432 / rate-limited)
The `medical_knowledge`/`policy_rights`/`migrant_health` agents fall back to the static knowledge listed below automatically — check your `TAVILY_API_KEY` at https://app.tavily.com if you want live results back.

### ChromaDB / seed_rag.py
These exist in the repo (`backend/memory/chroma_memory.py`, `backend/db/seed_rag.py`) but aren't called by the active pipeline — see the ChromaDB note under Database Services. Nothing to troubleshoot unless you've re-wired the tools to use it yourself.

---

## 📊 Static Fallback Knowledge

Used when Tavily is unavailable/rate-limited — hardcoded in the tool files, not a database:

`backend/tools/rag_retrieval_tool.py` (medical):
1. Diabetes Management
2. Hypertension (High Blood Pressure)
3. Cold and Flu
4. Fever Management
5. COVID-19 Guidance
6. Mental Health Resources in Germany
7. Vaccination in Germany
8. Chest Pain Assessment

`backend/tools/policy_rag_tool.py` (policy/rights):
1. Health Insurance in Germany (GKV/PKV)
2. Health Rights for Asylum Seekers (Asylbewerberleistungsgesetz)
3. Health Rights for EU Citizens
4. Rights for Uninsured Patients
5. Prescription Medications (Rezeptpflichtige Medikamente)
6. Krankenversicherung in Deutschland (DE)
7. Gesundheitsrechte für Asylsuchende (DE)

---

## 🤝 Integrations

- **Tavily** — Web search for current medical info
- **Nominatim/OSM** — Geocoding without API key
- **Google Maps** — Directions URLs (no API key needed for links)
- **VRR** — Verkehrsverbund Rhein-Ruhr transit info

---

## 📝 License

For internal/educational use. Medical information should be validated by licensed healthcare professionals before clinical use.

---

*Built for P4H (People for Health) Oberhausen — Helping everyone navigate healthcare regardless of language or background.*
