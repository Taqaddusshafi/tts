"""Audio normalization and WAV encoding helpers."""

from io import BytesIO
import wave

import numpy as np
from numpy.typing import NDArray


def to_mono_float32(samples: NDArray[np.floating] | list[float]) -> NDArray[np.float32]:
    """Convert model audio output into a mono float32 NumPy array."""

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError("Audio samples must be a one-dimensional mono signal.")
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def resample_linear(
    samples: NDArray[np.float32],
    source_rate: int,
    target_rate: int,
) -> NDArray[np.float32]:
    """Resample with deterministic linear interpolation when rates differ."""

    if source_rate == target_rate:
        return samples
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive integers.")
    if samples.size == 0:
        return samples

    duration = samples.size / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, samples.size - 1, num=samples.size)
    target_positions = np.linspace(0.0, samples.size - 1, num=target_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def encode_wav(samples: NDArray[np.float32], sample_rate: int) -> bytes:
    """Encode mono float samples in [-1, 1] as 16-bit PCM WAV bytes."""

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return buffer.getvalue()


def normalize_to_wav(
    samples: NDArray[np.floating] | list[float],
    source_rate: int,
    target_rate: int,
) -> bytes:
    """Normalize model output to the configured service WAV format."""

    mono = to_mono_float32(samples)
    resampled = resample_linear(mono, source_rate, target_rate)
    return encode_wav(resampled, target_rate)

