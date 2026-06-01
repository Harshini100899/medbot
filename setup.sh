#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  P4H MedBot — Oberhausen  |  setup.sh
#  One-command local setup: venv, deps, Docker services, seed, launch
# ═══════════════════════════════════════════════════════════════════════════════
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}▶ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; exit 1; }
hdr()  { echo -e "\n${BOLD}── $* ──${NC}"; }

hdr "P4H MedBot — Oberhausen Setup"

# ── 1. Python version check ───────────────────────────────────────────────────
PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
MAJOR=$(echo "$PYVER" | cut -d. -f1)
MINOR=$(echo "$PYVER" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
  err "Python 3.10+ required (found $PYVER). Install from https://python.org"
fi
log "Python $PYVER ✓"

# ── 2. Virtual environment ───────────────────────────────────────────────────
hdr "Virtual Environment"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment..."
  $PYTHON -m venv "$VENV_DIR"
else
  log "Virtual environment already exists"
fi

# Activate
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  err "Cannot activate venv. Try: rm -rf $VENV_DIR and re-run."
fi
log "Virtual environment activated"

# ── 3. Install dependencies ───────────────────────────────────────────────────
hdr "Installing Python dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q
log "Dependencies installed ✓"

# ── 4. .env file ─────────────────────────────────────────────────────────────
hdr "Configuration"
if [ ! -f ".env" ]; then
  cp .env.example .env
  log ".env created from .env.example"
  echo ""
  warn "IMPORTANT: Edit .env before starting:"
  warn "  • LLM_PROVIDER=ollama (default) — or openai / anthropic"
  warn "  • TAVILY_API_KEY=... (optional — enables web search fallback)"
  warn "  • OPENAI_API_KEY=... (if using OpenAI)"
  echo ""
else
  log ".env already exists"
fi

# ── 5. Data directories ───────────────────────────────────────────────────────
hdr "Data Directories"
mkdir -p data/chroma_db data/logs
log "data/chroma_db and data/logs created ✓"

# ── 6. Docker (Redis + MongoDB) ───────────────────────────────────────────────
hdr "Docker Services (Redis + MongoDB)"
if command -v docker &>/dev/null && command -v docker-compose &>/dev/null; then
  log "Starting Redis and MongoDB via Docker Compose..."
  docker-compose up -d redis mongodb
  log "Waiting for services to be healthy..."
  sleep 5
  log "Docker services started ✓"
elif command -v docker &>/dev/null; then
  log "Trying docker compose (v2)..."
  docker compose up -d redis mongodb 2>/dev/null && log "Docker services started ✓" || {
    warn "docker-compose not found. Starting Redis & MongoDB manually if available..."
    start_manual_services
  }
else
  warn "Docker not found. Starting Redis & MongoDB manually if installed..."
  # Try to start Redis manually
  if command -v redis-server &>/dev/null; then
    redis-server --daemonize yes --port 6379 2>/dev/null || true
    log "Redis started (manual)"
  else
    warn "Redis not found. Install Redis or Docker for full functionality."
    warn "App will run in degraded mode (no rate limiting / session cache)."
  fi
  # MongoDB
  if command -v mongod &>/dev/null; then
    mkdir -p data/mongodb
    mongod --fork --logpath data/logs/mongod.log --dbpath data/mongodb 2>/dev/null || true
    log "MongoDB started (manual)"
  else
    warn "MongoDB not found. App will run without persistence storage."
  fi
fi

# ── 7. Ollama model check ─────────────────────────────────────────────────────
hdr "LLM Setup"
LLM_PROVIDER=$(grep -E "^LLM_PROVIDER=" .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "ollama")

if [ "$LLM_PROVIDER" = "ollama" ]; then
  if command -v ollama &>/dev/null; then
    OLLAMA_MODEL=$(grep -E "^OLLAMA_MODEL=" .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "llama3.2:3b")
    log "Ollama found. Pulling $OLLAMA_MODEL (this may take a few minutes on first run)..."
    ollama pull "$OLLAMA_MODEL" || warn "Could not pull model. Start Ollama manually and run: ollama pull $OLLAMA_MODEL"
  else
    warn "Ollama not installed. Install from https://ollama.com"
    warn "After install, run: ollama pull llama3.2:3b"
    warn "Or switch LLM_PROVIDER in .env to 'openai' or 'anthropic'"
  fi
else
  log "LLM_PROVIDER=$LLM_PROVIDER — skipping Ollama setup"
fi

# ── 8. Seed ChromaDB ─────────────────────────────────────────────────────────
hdr "Seeding RAG Knowledge Base (ChromaDB)"
log "Seeding medical knowledge into ChromaDB..."
python -m backend.db.seed_rag && log "ChromaDB seeded ✓" || warn "Seed failed (app will still start; retry manually: python -m backend.db.seed_rag)"

# ── 9. Done ───────────────────────────────────────────────────────────────────
hdr "Setup Complete 🎉"
echo ""
echo -e "${BOLD}Start the server:${NC}"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo -e "${BOLD}Then open:${NC}  http://localhost:8000"
echo ""
echo -e "${BOLD}API docs:${NC}   http://localhost:8000/docs"
echo ""
