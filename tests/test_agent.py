import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from app.agent.graph import build_graph


@pytest.mark.asyncio
async def test_population_question():
    mock_country_data = {
        "name": {"common": "Germany", "official": "Federal Republic of Germany"},
        "population": 83_000_000,
        "capital": ["Berlin"],
        "currencies": {"EUR": {"name": "Euro"}},
        "languages": {"deu": "German"},
    }

    with patch("app.agent.tools.fetch_country", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_country_data
        graph = build_graph()
        result = await graph.ainvoke({"question": "What is the population of Germany?"})

    assert result["country"] == "Germany"
    assert "population" in result["fields_requested"]
    assert "83" in result["answer"] or "Germany" in result["answer"]


@pytest.mark.asyncio
async def test_unknown_country():
    with patch("app.agent.tools.fetch_country", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = None
        graph = build_graph()
        result = await graph.ainvoke({"question": "What is the capital of Narnia?"})

    assert result.get("tool_error") is not None or "not found" in result["answer"].lower()


@pytest.mark.asyncio
async def test_no_country_in_question():
    graph = build_graph()
    result = await graph.ainvoke({"question": "What is 2 + 2?"})
    assert result.get("intent_error") is not None or result["country"] is None