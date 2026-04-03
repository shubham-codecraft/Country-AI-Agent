from pydantic import BaseModel, Field, field_validator
from typing import Any
from app.core.config import get_settings


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question about a country",
        examples=["What is the population of Germany?"],
    )

    @field_validator("question")
    @classmethod
    def check_length(cls, v: str) -> str:
        settings = get_settings()
        if len(v) > settings.max_question_length:
            raise ValueError(
                f"Question exceeds max length of {settings.max_question_length} chars"
            )
        return v.strip()


class AskResponse(BaseModel):
    answer: str = Field(..., description="Natural language answer")
    country: str | None = Field(None, description="Detected country name")
    fields_requested: list[str] = Field(
        default_factory=list,
        description="Data fields the agent identified as needed",
    )
    raw_data: dict[str, Any] | None = Field(
        None, description="Subset of API data used to ground the answer"
    )
    error: str | None = Field(None, description="Error message if agent failed")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"