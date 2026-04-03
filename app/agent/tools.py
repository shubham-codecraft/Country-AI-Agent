"""
Thin async wrapper around the REST Countries API.

Design notes:
- Uses httpx.AsyncClient for non-blocking I/O inside FastAPI's async loop.
- Retry logic via tenacity handles transient network failures.
- Returns only the first match (most queries are unambiguous).
- Caller is responsible for picking the fields it needs.
"""

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Fields we actually surface — keeps payload small and avoids PII-adjacent data
ALLOWED_FIELDS = [
    "name",
    "capital",
    "population",
    "area",
    "region",
    "subregion",
    "languages",
    "currencies",
    "timezones",
    "flags",
    "borders",
    "continents",
    "tld",
    "latlng",
]

_FIELDS_PARAM = ",".join(ALLOWED_FIELDS)


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def fetch_country(country_name: str) -> dict[str, Any] | None:
    """
    Fetch country data from REST Countries API.

    Returns the first matching country dict, or None if not found.
    Raises httpx.HTTPStatusError on unexpected HTTP errors.
    """
    settings = get_settings()
    url = f"{settings.rest_countries_base_url}/name/{country_name}"
    params = {"fields": _FIELDS_PARAM, "fullText": "false"}

    logger.info("Fetching country data", extra={"country": country_name, "url": url})

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        response = await client.get(url, params=params)

        if response.status_code == 404:
            logger.warning("Country not found", extra={"country": country_name})
            return None

        response.raise_for_status()
        data = response.json()

    if not data:
        return None

    # API returns a list; take the best match (first result)
    country = data[0]
    logger.info(
        "Country data fetched",
        extra={"country": country_name, "official_name": country.get("name", {}).get("official")},
    )
    return country


def extract_fields(
    country_data: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    """
    Pull only the requested fields out of raw API data.

    Handles nested structures (name, currencies, languages) gracefully.
    """
    result: dict[str, Any] = {}

    field_map = {
        "name": lambda d: d.get("name", {}).get("common"),
        "official_name": lambda d: d.get("name", {}).get("official"),
        "capital": lambda d: d.get("capital", [None])[0],
        "population": lambda d: d.get("population"),
        "area": lambda d: d.get("area"),
        "region": lambda d: d.get("region"),
        "subregion": lambda d: d.get("subregion"),
        "languages": lambda d: list(d.get("languages", {}).values()),
        "currencies": lambda d: [
            f"{v.get('name')} ({k})"
            for k, v in d.get("currencies", {}).items()
        ],
        "timezones": lambda d: d.get("timezones", []),
        "borders": lambda d: d.get("borders", []),
        "continents": lambda d: d.get("continents", []),
        "tld": lambda d: d.get("tld", []),
        "flag": lambda d: d.get("flags", {}).get("png"),
    }

    for field in fields:
        extractor = field_map.get(field)
        if extractor:
            value = extractor(country_data)
            if value is not None:
                result[field] = value

    # Always include country name for grounding
    if "name" not in result:
        result["name"] = country_data.get("name", {}).get("common", "Unknown")

    return result