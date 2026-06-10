"""TTS service orchestration and guardrails."""

import re

from app.core.config import Settings
from app.core.errors import ValidationAppError
from app.core.languages import is_indic, normalize_language
from app.core.logger import get_logger
from app.models.loaders import BARK_MODEL_KEY, INDIC_MODEL_KEY
from app.models.registry import ModelRegistry

logger = get_logger(__name__)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


class TTSService:
    """Service layer for routing text to the correct TTS backend."""

    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Validate input, route to a model, and return normalized WAV bytes."""

        clean_text = self._sanitize_text(text)
        normalized_language = normalize_language(language)
        model_key = INDIC_MODEL_KEY if is_indic(normalized_language) else BARK_MODEL_KEY

        logger.info(
            "tts_request_routed",
            language=normalized_language,
            model=model_key,
            chars=len(clean_text),
            voice=voice,
        )

        model = self.registry.get_model(model_key)
        return model.synthesize(clean_text, normalized_language, voice)

    def _sanitize_text(self, text: str) -> str:
        """Remove control characters, collapse whitespace, and enforce limits."""

        clean_text = CONTROL_CHARS_RE.sub("", text)
        clean_text = WHITESPACE_RE.sub(" ", clean_text).strip()

        if not clean_text:
            raise ValidationAppError("Text must not be empty.")

        if len(clean_text) > self.settings.max_text_chars:
            raise ValidationAppError(
                "Text exceeds maximum length.",
                {"max_text_chars": self.settings.max_text_chars},
            )

        return clean_text

