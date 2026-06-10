"""Model registry wiring."""

from app.core.config import Settings
from app.models.registry import ModelRegistry
from app.models.tts_indic_parler import BarkTTSModel, IndicParlerTTSModel

INDIC_MODEL_KEY = "indic_parler"
BARK_MODEL_KEY = "bark"


def build_model_registry(settings: Settings) -> ModelRegistry:
    """Build the TTS model registry with lazy singleton loaders."""

    registry = ModelRegistry()
    registry.register_loader(INDIC_MODEL_KEY, lambda: IndicParlerTTSModel(settings))
    registry.register_loader(BARK_MODEL_KEY, lambda: BarkTTSModel(settings))
    return registry

