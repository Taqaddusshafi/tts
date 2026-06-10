"""TTS request and error schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.core.languages import normalize_language


class TTSRequest(BaseModel):
    """Request payload for POST /v1/tts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(..., min_length=1)
    language: str = Field(..., min_length=2, max_length=16)
    voice: str | None = Field(default=None, max_length=256)

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, value: str) -> str:
        """Apply configured text length cap at schema level."""

        max_text_chars = get_settings().max_text_chars
        if len(value.strip()) > max_text_chars:
            raise ValueError(f"text must be at most {max_text_chars} characters")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        """Validate and normalize language code."""

        try:
            return normalize_language(value)
        except Exception as exc:
            raise ValueError(str(exc)) from exc


class ErrorDetail(BaseModel):
    """Structured API error body."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Foundation-style error response envelope."""

    error: ErrorDetail
