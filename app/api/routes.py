import logging
from fastapi import APIRouter, HTTPException, status

from app.agent.graph import agent_graph
from app.models.schemas import AskRequest, AskResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["agent"],
    summary="Ask a question about a country",
    response_description="Natural language answer grounded in live API data",
)
async def ask(request: AskRequest) -> AskResponse:
    """
    Run the country-info agent on a natural language question.

    The agent:
    1. Identifies the country and requested data fields (intent node)
    2. Fetches live data from REST Countries API (tool node)
    3. Synthesizes a grounded answer (synthesis node)
    """
    logger.info("Received question", extra={"question": request.question})

    try:
        final_state = await agent_graph.ainvoke(
            {"question": request.question}
        )
    except Exception as exc:
        logger.exception("Agent graph raised unhandled exception: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent encountered an unexpected error. Please try again.",
        )

    return AskResponse(
        answer=final_state.get("answer", "No answer generated."),
        country=final_state.get("country"),
        fields_requested=final_state.get("fields_requested", []),
        raw_data=final_state.get("relevant_data"),
        error=final_state.get("intent_error") or final_state.get("tool_error"),
    )