"""Pytest fixtures for the TTS service."""

from collections.abc import Generator
import os

os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["WARMUP_ON_STARTUP"] = "false"
os.environ["MAX_TEXT_CHARS"] = "20"

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.models.loaders import BARK_MODEL_KEY, INDIC_MODEL_KEY
from app.models.registry import ModelRegistry
from app.services.tts_service import TTSService
from app.utils.audio_utils import encode_wav


class FakeTTSModel:
    """Small deterministic test double that returns a valid WAV payload."""

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls: list[tuple[str, str, str | None]] = []
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return whether the fake model was loaded."""

        return self.loaded

    def load(self) -> None:
        """Mark the fake model as loaded."""

        self.loaded = True

    def warmup(self) -> None:
        """Warmup is a no-op for fake models."""

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Return a tiny valid WAV with route metadata stored in calls."""

        self.calls.append((text, language, voice))
        return encode_wav([0.0, 0.1, -0.1, 0.0], 24000)


@pytest.fixture
def settings() -> Settings:
    """Return test settings with lightweight startup."""

    return Settings(load_models_on_startup=False, warmup_on_startup=False, max_text_chars=20)


@pytest.fixture
def fake_registry() -> ModelRegistry:
    """Return a registry wired to fake Indic and Bark models."""

    registry = ModelRegistry()
    registry.indic_model = FakeTTSModel("indic")  # type: ignore[attr-defined]
    registry.bark_model = FakeTTSModel("bark")  # type: ignore[attr-defined]
    registry.register_loader(INDIC_MODEL_KEY, lambda: registry.indic_model)  # type: ignore[attr-defined]
    registry.register_loader(BARK_MODEL_KEY, lambda: registry.bark_model)  # type: ignore[attr-defined]
    return registry


@pytest.fixture
def service(fake_registry: ModelRegistry, settings: Settings) -> TTSService:
    """Return a TTS service backed by fake models."""

    return TTSService(registry=fake_registry, settings=settings)


@pytest.fixture
def client(fake_registry: ModelRegistry, settings: Settings) -> Generator[TestClient, None, None]:
    """Return a test client with app state overridden to avoid real model loads."""

    with TestClient(app) as test_client:
        test_client.app.state.model_registry = fake_registry
        test_client.app.state.tts_service = TTSService(registry=fake_registry, settings=settings)
        yield test_client
