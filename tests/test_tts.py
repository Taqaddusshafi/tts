"""Tests for TTS routing, guardrails, and HTTP behavior."""

import wave
from io import BytesIO

import pytest

from app.core.errors import ValidationAppError
from app.models.loaders import BARK_MODEL_KEY, INDIC_MODEL_KEY
from app.models.registry import ModelRegistry
from app.services.tts_service import TTSService


def test_indic_text_routes_to_parler(
    service: TTSService,
    fake_registry: ModelRegistry,
) -> None:
    """Hindi requests should route to the Indic-Parler backend."""

    wav = service.synthesize("नमस्ते", "hi")

    assert wav.startswith(b"RIFF")
    assert fake_registry.is_loaded(INDIC_MODEL_KEY)
    assert not fake_registry.is_loaded(BARK_MODEL_KEY)


def test_non_indic_text_routes_to_bark(
    service: TTSService,
    fake_registry: ModelRegistry,
) -> None:
    """English requests should route to Bark."""

    wav = service.synthesize("Hello", "en")

    assert wav.startswith(b"RIFF")
    assert fake_registry.is_loaded(BARK_MODEL_KEY)
    assert not fake_registry.is_loaded(INDIC_MODEL_KEY)


def test_text_length_cap_rejected(service: TTSService) -> None:
    """Input beyond the configured cap should fail before inference."""

    with pytest.raises(ValidationAppError):
        service.synthesize("x" * 21, "en")


def test_bad_language_rejected(service: TTSService) -> None:
    """Unsupported language codes should fail validation."""

    with pytest.raises(ValidationAppError):
        service.synthesize("Hello", "zz")


@pytest.mark.parametrize(
    ("payload", "expected_model"),
    [
        ({"text": "नमस्ते", "language": "hi"}, INDIC_MODEL_KEY),
        ({"text": "Hello", "language": "en"}, BARK_MODEL_KEY),
    ],
)
def test_v1_tts_returns_valid_wav(client, fake_registry: ModelRegistry, payload: dict[str, str], expected_model: str) -> None:
    """POST /v1/tts should return a valid non-empty WAV for Hindi and English."""

    response = client.post("/v1/tts", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert fake_registry.is_loaded(expected_model)

    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() > 0


def test_v1_tts_rejects_bad_language(client) -> None:
    """The HTTP route should return the shared error envelope on validation errors."""

    response = client.post("/v1/tts", json={"text": "Hello", "language": "zz"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"

