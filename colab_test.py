# ═══════════════════════════════════════════════════════════════
# TTS Inference Service — Google Colab GPU Test Notebook
# ═══════════════════════════════════════════════════════════════
# Copy each "# --- CELL N ---" block into a separate Colab cell.
# Make sure to select: Runtime → Change runtime type → GPU (T4)
# ═══════════════════════════════════════════════════════════════


# --- CELL 1: Check GPU availability ---
# Run this first to confirm Colab gave you a GPU.

import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device      : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected! Go to Runtime → Change runtime type → GPU")


# --- CELL 2: Clone repo & install dependencies ---
# Replace the URL below with YOUR actual repo URL.

# !git clone https://github.com/YOUR_USERNAME/kdext_conversa_ai_tts.git
# %cd kdext_conversa_ai_tts

# OR if you want to upload files manually, skip the clone and upload
# the project as a zip, then:
# !unzip tts.zip -d tts && %cd tts

# Install dependencies
# !pip install -q -r requirements.txt


# --- CELL 3: Verify imports and configuration ---
# This checks that all modules load without errors.

import sys, os
os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["WARMUP_ON_STARTUP"] = "false"
os.environ["DEVICE"] = "auto"
os.environ["TORCH_DTYPE"] = "float16"

from app.core.config import get_settings
from app.core.languages import INDIC_LANGUAGES, SUPPORTED_NON_INDIC_LANGUAGES
from app.models.registry import ModelRegistry
from app.models.loaders import build_model_registry
from app.services.tts_service import TTSService

settings = get_settings()
print(f"✅ App name      : {settings.app_name}")
print(f"✅ Device config  : {settings.device}")
print(f"✅ Torch dtype    : {settings.torch_dtype}")
print(f"✅ Indic model    : {settings.indic_model_id}")
print(f"✅ Bark model     : {settings.bark_model_id}")
print(f"✅ Indic languages: {len(INDIC_LANGUAGES)} ({', '.join(sorted(INDIC_LANGUAGES)[:5])}...)")
print(f"✅ Non-Indic      : {len(SUPPORTED_NON_INDIC_LANGUAGES)} ({', '.join(sorted(SUPPORTED_NON_INDIC_LANGUAGES)[:5])}...)")


# --- CELL 4: Run unit tests (uses mock models, no GPU needed) ---
# This validates routing logic, validation, and HTTP behavior.

# !pytest tests/ -v --tb=short


# --- CELL 5: Load Indic-Parler-TTS model on GPU ---
# This downloads and loads the model. First run takes 2-5 min.

import time

# Clear any cached settings from Cell 3
get_settings.cache_clear()
os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["WARMUP_ON_STARTUP"] = "false"
os.environ["DEVICE"] = "auto"
os.environ["TORCH_DTYPE"] = "float16"

settings = get_settings()
registry = build_model_registry(settings)

print("⏳ Loading Indic-Parler-TTS model (this may take a few minutes on first run)...")
t0 = time.time()
indic_model = registry.get_model("indic_parler")
print(f"✅ Indic-Parler-TTS loaded in {time.time() - t0:.1f}s")
print(f"   Device: {indic_model.device}")
print(f"   Dtype : {indic_model.torch_dtype}")

if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"   GPU memory used: {mem_gb:.2f} GB")


# --- CELL 6: Test Indic TTS — Hindi ---

from IPython.display import Audio, display
import wave
from io import BytesIO

service = TTSService(registry=registry, settings=settings)

print("🔊 Synthesizing Hindi...")
t0 = time.time()
wav_bytes = service.synthesize("नमस्ते, मेरा नाम कन्वर्सा है।", "hi")
elapsed = time.time() - t0
print(f"✅ Hindi WAV generated in {elapsed:.2f}s ({len(wav_bytes)} bytes)")

# Parse and display audio info
with wave.open(BytesIO(wav_bytes), "rb") as wf:
    print(f"   Channels  : {wf.getnchannels()}")
    print(f"   Sample rate: {wf.getframerate()}")
    print(f"   Frames    : {wf.getnframes()}")
    duration = wf.getnframes() / wf.getframerate()
    print(f"   Duration  : {duration:.2f}s")

# Play audio in Colab
display(Audio(wav_bytes, rate=24000, autoplay=True))

# Save to file
with open("hindi_output.wav", "wb") as f:
    f.write(wav_bytes)
print("💾 Saved to hindi_output.wav")


# --- CELL 7: Test Indic TTS — Tamil ---

print("🔊 Synthesizing Tamil...")
t0 = time.time()
wav_bytes_ta = service.synthesize("வணக்கம், நான் கன்வர்சா.", "ta")
elapsed = time.time() - t0
print(f"✅ Tamil WAV generated in {elapsed:.2f}s ({len(wav_bytes_ta)} bytes)")
display(Audio(wav_bytes_ta, rate=24000, autoplay=True))


# --- CELL 8: Test Indic TTS — Bengali ---

print("🔊 Synthesizing Bengali...")
t0 = time.time()
wav_bytes_bn = service.synthesize("নমস্কার, আমি কনভার্সা।", "bn")
elapsed = time.time() - t0
print(f"✅ Bengali WAV generated in {elapsed:.2f}s ({len(wav_bytes_bn)} bytes)")
display(Audio(wav_bytes_bn, rate=24000, autoplay=True))


# --- CELL 9: Load Bark model & test English ---
# Bark handles non-Indic languages.

print("⏳ Loading Bark model...")
t0 = time.time()
bark_model = registry.get_model("bark")
print(f"✅ Bark loaded in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"   Total GPU memory used (both models): {mem_gb:.2f} GB")

print("\n🔊 Synthesizing English...")
t0 = time.time()
wav_bytes_en = service.synthesize(
    "Hello, this is a test of the Bark text to speech model.",
    "en",
    "v2/en_speaker_6",
)
elapsed = time.time() - t0
print(f"✅ English WAV generated in {elapsed:.2f}s ({len(wav_bytes_en)} bytes)")
display(Audio(wav_bytes_en, rate=24000, autoplay=True))

with open("english_output.wav", "wb") as f:
    f.write(wav_bytes_en)
print("💾 Saved to english_output.wav")


# --- CELL 10: Test voice descriptions (Indic-Parler custom voice) ---

print("🔊 Testing custom voice description for Hindi...")
t0 = time.time()
wav_custom = service.synthesize(
    "यह एक कस्टम वॉइस टेस्ट है।",
    "hi",
    "A soft, gentle female Indian voice with slow pacing and warm tone.",
)
elapsed = time.time() - t0
print(f"✅ Custom voice WAV in {elapsed:.2f}s")
display(Audio(wav_custom, rate=24000, autoplay=True))


# --- CELL 11: Start FastAPI server (background) + curl test ---
# This starts the actual HTTP server so you can test the API endpoint.

# import subprocess, time
# proc = subprocess.Popen(
#     ["python", "run.py"],
#     env={**os.environ, "LOAD_MODELS_ON_STARTUP": "true", "WARMUP_ON_STARTUP": "true"},
# )
# time.sleep(10)  # wait for models to load
# print("✅ Server started on port 8000")
#
# # Test health
# !curl -s http://localhost:8000/health | python -m json.tool
#
# # Test TTS
# !curl -s -X POST http://localhost:8000/v1/tts \
#   -H "Content-Type: application/json" \
#   -d '{"text":"नमस्ते","language":"hi"}' \
#   --output /content/test_api.wav
# print("✅ API test complete — saved to /content/test_api.wav")
# display(Audio("/content/test_api.wav", autoplay=True))
#
# # Kill server when done
# proc.terminate()


# --- CELL 12: GPU memory summary ---

if torch.cuda.is_available():
    print("═" * 50)
    print("GPU Memory Summary")
    print("═" * 50)
    print(f"Allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved  : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max alloc : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Device    : {torch.cuda.get_device_name(0)}")
    print("═" * 50)
else:
    print("Running on CPU — no GPU memory stats available.")
