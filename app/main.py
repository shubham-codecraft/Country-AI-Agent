import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Graph is compiled at import time in graph.py
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        await client.models.list()
        logger.info("OpenAI API connectivity verified")
    finally:
        await client.close()

    yield

    # Graceful shutdown hooks go here
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Country Information AI Agent",
        description="LangGraph-powered agent that answers questions about countries using live data.",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)