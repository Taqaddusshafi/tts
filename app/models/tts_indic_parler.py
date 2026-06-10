"""Indic-Parler-TTS and Bark model wrappers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.errors import InferenceError, ModelLoadError
from app.core.logger import get_logger
from app.utils.audio_utils import normalize_to_wav

logger = get_logger(__name__)


@dataclass(slots=True)
class GeneratedAudio:
    """Raw model audio before service-level normalization."""

    samples: np.ndarray
    sample_rate: int


class TorchRuntimeMixin:
    """Shared PyTorch runtime helpers for GPU/CPU model wrappers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cpu"
        self.torch_dtype: Any = None

    def _configure_torch(self) -> Any:
        """Import torch lazily and resolve device and dtype."""

        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise ModelLoadError("PyTorch is required for TTS inference.") from exc

        if self.settings.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.settings.device

        if self.settings.torch_dtype == "auto":
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif self.settings.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.settings.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        return torch


class IndicParlerTTSModel(TorchRuntimeMixin):
    """Wrapper around Indic-Parler-TTS for Indian languages."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._sample_rate = settings.output_sample_rate

    @property
    def is_loaded(self) -> bool:
        """Return whether the Parler model and tokenizer are ready."""

        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load Indic-Parler-TTS into memory once."""

        if self.is_loaded:
            return

        torch = self._configure_torch()
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            cache_dir = str(Path(self.settings.model_store_dir).resolve())
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.indic_model_id,
                cache_dir=cache_dir,
            )
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.settings.indic_model_id,
                cache_dir=cache_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self._sample_rate = int(getattr(self.model.config, "sampling_rate", self._sample_rate))
            logger.info(
                "indic_parler_loaded",
                model_id=self.settings.indic_model_id,
                device=self.device,
                dtype=str(self.torch_dtype),
                sample_rate=self._sample_rate,
            )
        except Exception as exc:
            raise ModelLoadError("Failed to load Indic-Parler-TTS model.") from exc

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """Run a short Hindi warmup utterance to reduce first request latency."""

        self.synthesize("नमस्ते", "hi", self.settings.default_indic_voice)

    def _generate_audio(self, text: str, voice: str | None = None) -> GeneratedAudio:
        """Generate raw audio with Indic-Parler-TTS."""

        if not self.is_loaded:
            self.load()

        assert self.model is not None
        assert self.tokenizer is not None

        description = voice or self.settings.default_indic_voice
        try:
            import torch

            description_inputs = self.tokenizer(
                description,
                return_tensors="pt",
            ).to(self.device)
            prompt_inputs = self.tokenizer(
                text,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                generation = self.model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )

            samples = generation.detach().cpu().numpy().squeeze().astype(np.float32)
            return GeneratedAudio(samples=samples, sample_rate=self._sample_rate)
        except Exception as exc:
            raise InferenceError("Indic-Parler-TTS inference failed.") from exc

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Generate normalized WAV bytes for Indic text."""

        audio = self._generate_audio(text=text, voice=voice)
        return normalize_to_wav(
            samples=audio.samples,
            source_rate=audio.sample_rate,
            target_rate=self.settings.output_sample_rate,
        )


class BarkTTSModel(TorchRuntimeMixin):
    """Wrapper around Bark for non-Indic languages."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model: Any | None = None
        self.processor: Any | None = None
        self._sample_rate = settings.output_sample_rate

    @property
    def is_loaded(self) -> bool:
        """Return whether Bark model resources are ready."""

        return self.model is not None and self.processor is not None

    def load(self) -> None:
        """Load Bark into memory once."""

        if self.is_loaded:
            return

        self._configure_torch()
        try:
            from transformers import AutoProcessor, BarkModel

            cache_dir = str(Path(self.settings.model_store_dir).resolve())
            self.processor = AutoProcessor.from_pretrained(
                self.settings.bark_model_id,
                cache_dir=cache_dir,
            )
            self.model = BarkModel.from_pretrained(
                self.settings.bark_model_id,
                cache_dir=cache_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self._sample_rate = int(getattr(self.model.generation_config, "sample_rate", 24000))
            logger.info(
                "bark_loaded",
                model_id=self.settings.bark_model_id,
                device=self.device,
                dtype=str(self.torch_dtype),
                sample_rate=self._sample_rate,
            )
        except Exception as exc:
            raise ModelLoadError("Failed to load Bark model.") from exc

    def warmup(self) -> None:
        """Run a short English warmup utterance to reduce first request latency."""

        self.synthesize("Hello.", "en", self.settings.default_bark_voice)

    def _generate_audio(self, text: str, voice: str | None = None) -> GeneratedAudio:
        """Generate raw audio with Bark."""

        if not self.is_loaded:
            self.load()

        assert self.model is not None
        assert self.processor is not None

        try:
            import torch

            voice_preset = voice or self.settings.default_bark_voice
            inputs = self.processor(
                text,
                voice_preset=voice_preset,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                generation = self.model.generate(**inputs)

            samples = generation.detach().cpu().numpy().squeeze().astype(np.float32)
            return GeneratedAudio(samples=samples, sample_rate=self._sample_rate)
        except Exception as exc:
            raise InferenceError("Bark inference failed.") from exc

    def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Generate normalized WAV bytes for non-Indic text."""

        audio = self._generate_audio(text=text, voice=voice)
        return normalize_to_wav(
            samples=audio.samples,
            source_rate=audio.sample_rate,
            target_rate=self.settings.output_sample_rate,
        )

