from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    total=False means every key is optional — nodes only write
    what they produce, keeping coupling loose.
    """

    # Input
    question: str                          # original user question

    # After intent node
    country: str | None                    # e.g. "Germany"
    fields_requested: list[str]            # e.g. ["population", "capital"]
    intent_error: str | None               # set if LLM couldn't parse intent

    # After tool node
    raw_country_data: dict[str, Any] | None   # full API response for the country
    relevant_data: dict[str, Any] | None      # filtered subset for requested fields
    tool_error: str | None                    # set if API call failed

    # After synthesis node
    answer: str                            # final natural language answer