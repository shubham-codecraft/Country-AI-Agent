"""
LangGraph node functions.

Each node receives the full AgentState, does one job, and returns
a partial state dict that LangGraph merges in.

Node 1 — intent:    Parse country name + requested fields from user question.
Node 2 — tool:      Call REST Countries API with the parsed country name.
Node 3 — synthesis: Generate a grounded natural language answer.
"""

import json
import logging
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.tools import fetch_country, extract_fields
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# --------------------------------------------------------------------------- #
# Shared LLM instance (stateless, safe to reuse across requests)
# --------------------------------------------------------------------------- #

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    api_key=settings.openai_api_key,
)

# --------------------------------------------------------------------------- #
# Node 1: Intent / Field Identification
# --------------------------------------------------------------------------- #

_INTENT_SYSTEM = """You are an intent parser for a country-information service.

Given a user question, extract:
1. "country": the country being asked about (use its common English name).
2. "fields": a list of data fields needed to answer. Choose ONLY from:
   [name, capital, population, area, region, subregion, languages,
    currencies, timezones, borders, continents, tld, flag]

Respond with ONLY valid JSON, no markdown, no explanation.
Example output: {"country": "Germany", "fields": ["population", "capital"]}

If you cannot identify a country, set country to null and fields to [].
"""


async def intent_node(state: AgentState) -> dict[str, Any]:
    """Extract country + fields from the natural language question."""
    question = state["question"]
    logger.info("Intent node running", extra={"question": question})

    try:
        response = await _llm.ainvoke(
            [
                SystemMessage(content=_INTENT_SYSTEM),
                HumanMessage(content=question),
            ]
        )
        raw = response.content.strip()
        parsed = json.loads(raw)

        country = parsed.get("country")
        fields = parsed.get("fields", [])

        if not country:
            logger.warning("Intent node could not identify a country")
            return {
                "country": None,
                "fields_requested": [],
                "intent_error": "I couldn't identify a country in your question. Please mention a specific country.",
            }

        logger.info("Intent parsed", extra={"country": country, "fields": fields})
        return {
            "country": country,
            "fields_requested": fields or ["name"],  # fallback: at least name
            "intent_error": None,
        }

    except (json.JSONDecodeError, Exception) as exc:
        logger.exception("Intent node failed: %s", exc)
        return {
            "country": None,
            "fields_requested": [],
            "intent_error": "Failed to parse your question. Please rephrase it.",
        }


# --------------------------------------------------------------------------- #
# Node 2: Tool Invocation
# --------------------------------------------------------------------------- #

async def tool_node(state: AgentState) -> dict[str, Any]:
    """Fetch country data from REST Countries API."""
    # Short-circuit if intent already failed
    if state.get("intent_error"):
        return {"raw_country_data": None, "relevant_data": None, "tool_error": None}

    country = state["country"]
    fields = state.get("fields_requested", [])

    logger.info("Tool node running", extra={"country": country, "fields": fields})

    try:
        raw_data = await fetch_country(country)

        if raw_data is None:
            return {
                "raw_country_data": None,
                "relevant_data": None,
                "tool_error": f"No data found for '{country}'. Please check the country name.",
            }

        relevant = extract_fields(raw_data, fields)
        logger.info("Tool node success", extra={"country": country, "extracted_keys": list(relevant.keys())})
        return {
            "raw_country_data": raw_data,
            "relevant_data": relevant,
            "tool_error": None,
        }

    except httpx.HTTPStatusError as exc:
        logger.error("HTTP error fetching country: %s", exc)
        return {
            "raw_country_data": None,
            "relevant_data": None,
            "tool_error": f"External API error ({exc.response.status_code}). Please try again later.",
        }
    except httpx.TransportError as exc:
        logger.error("Network error fetching country: %s", exc)
        return {
            "raw_country_data": None,
            "relevant_data": None,
            "tool_error": "Network error reaching the countries API. Please try again.",
        }


# --------------------------------------------------------------------------- #
# Node 3: Answer Synthesis
# --------------------------------------------------------------------------- #

_SYNTHESIS_SYSTEM = """You are a helpful assistant that answers questions about countries.
You will receive:
- The user's original question
- Retrieved country data (JSON)

Rules:
- Answer ONLY from the provided data. Do not hallucinate.
- Be concise but complete.
- If data for a specific field is missing, say so honestly.
- Format numbers with commas for readability (e.g., 1,000,000).
- Do not mention the API or internal workings.
"""


async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Synthesize a natural language answer from retrieved data."""
    # --- Error pass-through ---
    if state.get("intent_error"):
        return {"answer": state["intent_error"]}

    if state.get("tool_error"):
        return {"answer": state["tool_error"]}

    question = state["question"]
    relevant_data = state.get("relevant_data", {})

    if not relevant_data:
        return {"answer": "I found the country but couldn't retrieve the requested information."}

    logger.info("Synthesis node running", extra={"country": state.get("country")})

    data_str = json.dumps(relevant_data, indent=2)
    prompt = f"Question: {question}\n\nCountry data:\n{data_str}"

    try:
        response = await _llm.ainvoke(
            [
                SystemMessage(content=_SYNTHESIS_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        answer = response.content.strip()
        logger.info("Synthesis node complete", extra={"answer_length": len(answer)})
        return {"answer": answer}

    except Exception as exc:
        logger.exception("Synthesis node failed: %s", exc)
        return {"answer": "I retrieved the data but encountered an error generating the answer."}