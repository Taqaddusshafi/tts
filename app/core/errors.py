"""Application error types and HTTP exception mapping."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ErrorEnvelope:
    """Standard error response payload."""

    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope into the API error shape."""

        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class AppError(Exception):
    """Base service exception that can be rendered as an API error."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ValidationAppError(AppError):
    """Raised when request data is semantically invalid."""

    status_code = 422
    code = "validation_error"


class ModelLoadError(AppError):
    """Raised when a TTS model cannot be loaded."""

    status_code = 503
    code = "model_load_error"


class InferenceError(AppError):
    """Raised when model inference fails."""

    status_code = 500
    code = "inference_error"

