"""
tests/test_scraper.py — Tests for the arzt-auskunft.de scraper
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch


# ─── Scraper tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scraper_returns_list():
    """Verify scrape_arzt_auskunft returns a list (even if empty)."""
    from backend.tools.arzt_auskunft_scraper import scrape_arzt_auskunft
    # We mock httpx so tests don't need network
    mock_html = b"""
<!DOCTYPE html>
<html>
<body>
<div class="card card-hover mb-4" data-href-mobile="https://www.arzt-auskunft.de/arzt/innere-medizin/oberhausen/test-doctor-1">
    <div class="card-body" itemscope itemtype="https://schema.org/Physician">
        <h2 class="mb-0 h3" itemprop="name">Dr. Test Doctor</h2>
        <span class="text-subdued" itemprop="medicalSpecialty">Facharzt f&#252;r Innere Medizin und Endokrinologie und Diabetologie</span>
        <ul class="list-unstyled">
            <li itemprop="address" itemscope itemtype="https://schema.org/PostalAddress">
                <span itemprop="streetAddress">Teststra&#223;e 1</span>
                <span itemprop="postalCode">46045</span>
                <span itemprop="addressLocality">Oberhausen</span>
            </li>
        </ul>
    </div>
</div>
</body>
</html>
"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = mock_html

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("backend.tools.arzt_auskunft_scraper.httpx.AsyncClient", return_value=mock_client):
        result = await scrape_arzt_auskunft("diabetes", city="Oberhausen", limit=5)

    assert isinstance(result, list)
    # Should have parsed at least one doctor
    assert len(result) >= 1
    assert result[0]["name"] == "Dr. Test Doctor"
    assert "Oberhausen" in result[0]["address"]


@pytest.mark.asyncio
async def test_scraper_falls_back_on_network_error():
    """Verify scraper returns empty list on network error (no crash)."""
    from backend.tools.arzt_auskunft_scraper import scrape_arzt_auskunft
    import httpx

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Network error"))

    with patch("backend.tools.arzt_auskunft_scraper.httpx.AsyncClient", return_value=mock_client):
        result = await scrape_arzt_auskunft("diabetes", city="Oberhausen")

    assert result == []


def test_intent_routing_diabetes_doctor():
    """Verify 'i have a diabetes find a suitable doctor' routes to doctor_search."""
    from backend.agents.supervisor_agent import rule_based_intent
    text = "i have a diabetes find a suitable doctor in oberhausen"
    intent = rule_based_intent(text)
    assert intent == "doctor_search", f"Expected doctor_search but got: {intent}"


def test_intent_routing_find_doctor():
    """Verify explicit 'find me a doctor' routes to doctor_search."""
    from backend.agents.supervisor_agent import rule_based_intent
    intent = rule_based_intent("find me a doctor for heart problems")
    assert intent == "doctor_search"


def test_intent_routing_emergency():
    """Verify emergency keywords take priority."""
    from backend.agents.supervisor_agent import rule_based_intent
    intent = rule_based_intent("I am having a heart attack call 112")
    assert intent == "emergency"


def test_intent_routing_general_medical():
    """Verify a pure medical question does not misroute to doctor_search."""
    from backend.agents.supervisor_agent import rule_based_intent
    intent = rule_based_intent("what are the symptoms of flu?")
    # Should be None (goes to LLM) or medical_knowledge, not doctor_search
    assert intent != "doctor_search"


@pytest.mark.asyncio
async def test_find_doctors_uses_fallback_without_scraper():
    """Verify find_doctors returns fallback doctors when scraper and Tavily both fail."""
    from backend.tools.doctor_search_tool import find_doctors, FALLBACK_DOCTORS

    # Mock scraper to return empty (simulating network unavailability)
    with patch("backend.tools.doctor_search_tool.scrape_arzt_auskunft",
               new_callable=AsyncMock, return_value=[]), \
         patch("backend.tools.doctor_search_tool._tavily_doctor_search",
               new_callable=AsyncMock, return_value=[]):
        result = await find_doctors("diabetes", city="Oberhausen")

    assert "doctors" in result
    assert isinstance(result["doctors"], list)
    assert len(result["doctors"]) > 0
    # Should filter to diabetology specialists from fallback
    assert any("Diabet" in d.get("specialization", "") or
               "diabet" in d.get("specialization", "").lower()
               for d in result["doctors"])
