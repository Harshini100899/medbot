import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Mock database connections during startup imports
with patch("backend.memory.redis_memory.get_redis", return_value=None), \
     patch("backend.db.mongodb.get_db", return_value=None), \
     patch("backend.memory.chroma_memory.get_chroma_client", return_value=None):
    from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint returns correct structure and values."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert "redis" in data["services"]
    assert "mongodb" in data["services"]

def test_info_endpoint():
    """Test the application info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "agents" in data

def test_emergency_contacts():
    """Test emergency contacts endpoint."""
    response = client.get("/api/emergency/contacts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) or isinstance(data, dict)

@pytest.mark.asyncio
async def test_doctor_search_mock():
    """Test searching doctors endpoint — validates response structure.
    
    The /api/doctors/search endpoint delegates to find_doctors() which returns
    {doctors, hospitals, inferred_specialisation, city}. When MongoDB is
    unavailable the subagent falls back to FALLBACK_DOCTORS data.
    """
    response = client.get("/api/doctors/search?query=cardiologist&city=Oberhausen")
    assert response.status_code == 200
    data = response.json()
    # Response is a dict with these top-level keys
    assert "doctors" in data
    assert "hospitals" in data
    assert "inferred_specialisation" in data
    assert data["city"] == "Oberhausen"
    # Doctors list should be non-empty (fallback data is used when DB is down)
    assert isinstance(data["doctors"], list)
    # Each doctor entry must contain required fields
    if data["doctors"]:
        first = data["doctors"][0]
        assert "name" in first
        assert "specialization" in first

@pytest.mark.asyncio
async def test_web_search_tool_mock():
    """Test Tavily web search tool fallback logic."""
    from backend.tools.web_search_tool import medical_web_search, policy_web_search
    
    mock_results = [{"title": "Test Title", "url": "http://test.com", "content": "Test content snippet", "score": 0.99}]
    
    with patch("backend.tools.web_search_tool.web_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results
        
        # Test medical search
        med_res = await medical_web_search("fever", language="de")
        assert len(med_res) == 1
        assert med_res[0]["title"] == "Test Title"
        
        # Test policy search
        pol_res = await policy_web_search("insurance", language="en")
        assert len(pol_res) == 1
        assert pol_res[0]["url"] == "http://test.com"

@pytest.mark.asyncio
async def test_chat_message_flow_rule_based():
    """Test routing and processing using keyword rule-based intents."""
    with patch("backend.memory.redis_memory.check_rate_limit", new_callable=AsyncMock, return_value=True), \
         patch("backend.memory.redis_memory.get_messages", new_callable=AsyncMock, return_value=[]), \
         patch("backend.memory.redis_memory.push_message", new_callable=AsyncMock, return_value=None), \
         patch("backend.db.mongodb.create_session", new_callable=AsyncMock, return_value=None), \
         patch("backend.db.mongodb.save_conversation_turn", new_callable=AsyncMock, return_value=None):
         
        # Send a message with emergency keyword "112"
        response = client.post("/api/chat/message", json={"message": "Help me 112"})
        assert response.status_code == 200
        data = response.json()
        assert data["is_emergency"] is True
        assert data["intent"] == "emergency"
        assert "112" in data["response"]

@pytest.mark.asyncio
async def test_chat_message_flow_llm():
    """Test full chat message flow using a mocked process_message call.

    Python's `from module import fn` creates local bindings, so patching
    deep dependencies (redis, LLM, mongodb) at their source modules does NOT
    intercept calls from supervisor_agent / supervisor_graph which have already
    bound local references.

    Instead we mock `process_message` at the router's import site
    (`backend.api.chat_router.process_message`). This correctly verifies that:
    - The HTTP endpoint calls process_message with the user message
    - The ChatResponse model is populated from the returned dict
    - Intent, response text and is_emergency are mapped correctly
    """
    mock_result = {
        "response": "This is a mocked response about cold.",
        "session_id": "test-session-123",
        "language": "en",
        "intent": "medical_knowledge",
        "agent": "medical_knowledge_agent",
        "is_emergency": False,
        "sources": [],
        "metadata": {},
    }

    with patch("backend.api.chat_router.check_rate_limit", new_callable=AsyncMock, return_value=True), \
         patch("backend.api.chat_router.process_message", new_callable=AsyncMock, return_value=mock_result):

        response = client.post("/api/chat/message", json={"message": "I have cold symptoms"})
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "medical_knowledge"
        assert data["is_emergency"] is False
        assert "mocked response" in data["response"]
        assert data["session_id"] == "test-session-123"
