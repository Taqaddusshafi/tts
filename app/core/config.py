"""Environment-driven application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "tts-inference-service"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    model_store_dir: Path = Field(default=Path("model_store"))
    indic_model_id: str = "ai4bharat/indic-parler-tts"
    bark_model_id: str = "suno/bark-small"
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "float16"
    load_models_on_startup: bool = True
    warmup_on_startup: bool = True

    max_text_chars: int = Field(default=1000, ge=1, le=10000)
    output_sample_rate: int = Field(default=44100, ge=8000, le=48000)

    default_indic_voice: str = (
        "A clear, natural Indian voice with neutral pacing and studio quality."
    )
    default_bark_voice: str = "v2/en_speaker_6"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so all modules share the same configuration."""

    return Settings()

