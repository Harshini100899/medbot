# P4H MedBot — Oberhausen
**Multilingual Medical AI Chatbot** · DE | EN | TR | UK

A full-stack, production-ready medical chatbot built with **LangGraph multi-agent architecture**, FastAPI, Redis, MongoDB, and ChromaDB. Designed for Oberhausen residents — especially migrants, refugees, and non-German speakers — to navigate the German healthcare system.

---

## 🏗 Architecture

```
User (DE|EN|TR|UK)
        │
        ▼
  FastAPI Gateway ─── SSE Streaming ──► Browser
        │
  Language Processing
  (langdetect + multilingual embeddings)
        │
  Medical Ontology Normalizer
  (SNOMED-CT / ICD-10-GM / MeSH)
        │
  ┌─────▼──────────────────────────────────────┐
  │   LEVEL 1 — SUPERVISOR AGENT (LangGraph)   │
  │   Intent Classifier · ReACT · Delegation   │
  └────────┬───────────────────────────────────┘
           │ routes to one of six:
    ┌──────▼──────────────────────────────────────────┐
    │  LEVEL 2 — SIX SPECIALIST AGENTS               │
    │  Emergency · Doctor Search · Medical Knowledge │
    │  Policy & Rights · Location & Maps             │
    │  Migrant & Refugee Health                      │
    └──────┬──────────────────────────────────────────┘
           │ uses:
    ┌──────▼──────────────────────────────────────────┐
    │  LEVEL 3 — SUB-AGENTS                          │
    │  RAG Retrieval · Doctor Search · Policy RAG    │
    │  Maps Sub-agent                                │
    └──────┬──────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │  DATA LAYER                                     │
    │  ChromaDB (vectors) · Redis (short-term)        │
    │  MongoDB (long-term) · Tavily (web search)      │
    └─────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Languages** | German, English, Turkish, Ukrainian (auto-detected) |
| **6 Specialist Agents** | Emergency, Doctor Search, Medical Knowledge, Policy & Rights, Location & Maps, Migrant Health |
| **RAG** | ChromaDB with multilingual sentence embeddings + Tavily web fallback |
| **Short-term Memory** | Redis (session context, rate limiting, 3600s TTL) |
| **Long-term Memory** | MongoDB (conversation history, sessions) |
| **Streaming** | Server-Sent Events (SSE) — real-time token-by-token response |
| **Ontology** | SNOMED-CT, ICD-10-GM, MeSH local normalizer |
| **Emergency Fast-path** | Instant 112 banner — bypasses all LLM calls |
| **Graceful degradation** | Works without Redis/MongoDB/Tavily (reduced features) |
| **LLM Agnostic** | Ollama (local) · OpenAI · Anthropic — switch via `.env` |

---

## 📋 Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required |
| Docker + Docker Compose | Any | For Redis & MongoDB |
| Ollama | Latest | For local LLM (default) |
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
- Seed ChromaDB with medical knowledge

### 3. Edit `.env` (optional but recommended)
```bash
nano .env   # or any editor
```

Key settings:
```env
LLM_PROVIDER=ollama          # ollama | openai | anthropic
OLLAMA_MODEL=medgemma        # Google MedGemma — medical-tuned (recommended)
TAVILY_API_KEY=tvly-...      # get free at https://tavily.com (optional)
OPENAI_API_KEY=sk-...        # only if using OpenAI instead of Ollama
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
│   ├── llm_factory.py           # Ollama/OpenAI/Anthropic factory
│   ├── agents/
│   │   ├── supervisor_agent.py  # Intent classification + routing
│   │   ├── emergency_agent.py   # 🚨 Fast-path 112 response
│   │   ├── doctor_search_agent.py
│   │   ├── medical_knowledge_agent.py
│   │   ├── policy_rights_agent.py
│   │   ├── location_maps_agent.py
│   │   └── migrant_health_agent.py
│   ├── subagents/
│   │   ├── rag_retrieval_subagent.py
│   │   ├── doctor_search_subagent.py
│   │   ├── policy_rag_subagent.py
│   │   └── maps_subagent.py
│   ├── graph/
│   │   ├── state.py             # LangGraph state definition
│   │   └── supervisor_graph.py  # Main LangGraph StateGraph
│   ├── memory/
│   │   ├── redis_memory.py      # Short-term + rate limiting
│   │   └── chroma_memory.py     # Vector store operations
│   ├── db/
│   │   ├── mongodb.py           # Motor async MongoDB client
│   │   └── seed_rag.py          # Seeds ChromaDB knowledge base
│   ├── language/
│   │   ├── detector.py          # langdetect + DE/EN/TR/UK logic
│   │   └── embeddings.py        # sentence-transformers wrapper
│   ├── ontology/
│   │   └── normalizer.py        # SNOMED-CT / ICD-10-GM / MeSH
│   ├── response_builder/
│   │   └── builder.py           # Assembles final response + disclaimers
│   ├── tools/
│   │   ├── web_search_tool.py   # Tavily async search
│   │   └── maps_tool.py         # Google Maps + Nominatim geocoding
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
  "intent": "medical_knowledge",
  "agent": "medical_knowledge",
  "is_emergency": false,
  "sources": ["ChromaDB: medical_knowledge"],
  "metadata": {}
}
```

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

---

## 🔍 Tavily Web Search (Optional)
Get a free API key at https://tavily.com (1000 searches/month free).
```env
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```
Without it, the bot falls back to local ChromaDB knowledge only.

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

### ChromaDB (Vector store)
- Medical knowledge (8 topics seeded)
- Policy & rights documents (2 seeded)
- Session summaries
- Multilingual cosine similarity search

---

## 🌍 Multilingual Support

| Language | Code | Detection | Embeddings |
|---|---|---|---|
| Deutsch | `de` | ✅ | ✅ |
| English | `en` | ✅ | ✅ |
| Türkçe | `tr` | ✅ | ✅ |
| Українська | `uk` | ✅ (custom) | ✅ |

The embeddings model (`paraphrase-multilingual-MiniLM-L12-v2`) handles all four languages natively for cross-lingual RAG retrieval.

---

## 🚨 Emergency Handling

Any message containing emergency keywords in DE/EN/TR/UK triggers the **Emergency Fast-path**:
- Instant response with 112 banner
- First-aid LLM guidance
- Nearest hospitals in Oberhausen
- Full emergency number table

Keywords: `help`, `emergency`, `notruf`, `ambulanz`, `acil`, `допоможіть`, etc.

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

### ChromaDB errors
```bash
rm -rf data/chroma_db
python -m backend.db.seed_rag
```

### Re-seed knowledge base
```bash
source .venv/bin/activate
python -m backend.db.seed_rag
```

---

## 📊 Knowledge Base (Seeded)

Medical topics in ChromaDB:
1. Cold & Flu
2. Fever management
3. Chest pain assessment
4. Hypertension
5. Diabetes management
6. Mental health resources
7. COVID-19 guidance
8. Vaccination schedule (Germany)

Policy topics:
1. GKV health insurance coverage
2. Healthcare rights for asylum seekers (AsylbLG)

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
