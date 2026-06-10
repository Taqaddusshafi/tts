"""Thread-safe lazy model registry."""

from collections.abc import Callable
from threading import RLock
from typing import Protocol

from app.core.errors import ModelLoadError
from app.core.logger import get_logger

logger = get_logger(__name__)


class TTSModel(Protocol):
    """Common interface implemented by TTS model wrappers."""

    @property
    def is_loaded(self) -> bool:
        """Return whether the wrapper has loaded its underlying model."""

    def load(self) -> None:
        """Load model resources into memory."""

    def warmup(self) -> None:
        """Run a small warmup inference."""

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Generate WAV bytes for text."""


ModelLoader = Callable[[], TTSModel]


class ModelRegistry:
    """Lazy singleton registry for model wrappers."""

    def __init__(self) -> None:
        self._loaders: dict[str, ModelLoader] = {}
        self._models: dict[str, TTSModel] = {}
        self._lock = RLock()

    def register_loader(self, name: str, loader: ModelLoader) -> None:
        """Register a lazy model factory."""

        with self._lock:
            self._loaders[name] = loader

    def get_model(self, name: str) -> TTSModel:
        """Return a loaded singleton model by name."""

        with self._lock:
            if name in self._models:
                return self._models[name]

            loader = self._loaders.get(name)
            if loader is None:
                raise ModelLoadError("Model loader is not registered.", {"model": name})

            logger.info("model_load_start", model=name)
            model = loader()
            model.load()
            self._models[name] = model
            logger.info("model_load_complete", model=name)
            return model

    def preload(self, names: list[str] | None = None) -> None:
        """Load selected models, or all registered models, into memory."""

        selected = names or list(self._loaders.keys())
        for name in selected:
            self.get_model(name)

    def warmup(self, names: list[str] | None = None) -> None:
        """Warm selected loaded models."""

        selected = names or list(self._loaders.keys())
        for name in selected:
            model = self.get_model(name)
            logger.info("model_warmup_start", model=name)
            model.warmup()
            logger.info("model_warmup_complete", model=name)

    def is_loaded(self, name: str) -> bool:
        """Return whether a named model has been instantiated and loaded."""

        model = self._models.get(name)
        return bool(model and model.is_loaded)

    def any_loaded(self) -> bool:
        """Return whether any registered model has loaded successfully."""

        return any(model.is_loaded for model in self._models.values())

    def registered_names(self) -> list[str]:
        """Return registered model names."""

        return list(self._loaders.keys())

