# 🚀 TTS Colab GPU Test Guide

Complete step-by-step guide to push your project to GitHub and test it on Google Colab with GPU.

---

## Part 1 — Push to GitHub

### Step 1: Create a GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. **Repository name**: `kdext_conversa_ai_tts`
3. **Visibility**: Private (or Public)
4. **Do NOT** initialize with README (you already have one)
5. Click **Create repository**

### Step 2: Push your local code

Open your terminal and run these commands one by one:

```bash
cd /Users/taqaddusshafi/Desktop/tts

git init
git add .
git commit -m "Initial commit: TTS inference service"

git remote add origin https://github.com/YOUR_USERNAME/kdext_conversa_ai_tts.git
git branch -M main
git push -u origin main
```

> ⚠️ Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 3: Verify

Go to `https://github.com/YOUR_USERNAME/kdext_conversa_ai_tts` and confirm all files are uploaded.

---

## Part 2 — Google Colab Setup

### Step 4: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook**
3. Go to **Runtime → Change runtime type**
4. Select **GPU** (T4 is free and works great)
5. Click **Save**

### Step 5: Verify GPU (Cell 1)

Paste this in the **first cell** and run it:

```python
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device      : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️  No GPU! Go to Runtime → Change runtime type → GPU")
```

✅ You should see `CUDA available: True` and a GPU name like `Tesla T4`.

---

## Part 3 — Clone & Install

### Step 6: Clone your repo (Cell 2)

```python
!git clone https://github.com/YOUR_USERNAME/kdext_conversa_ai_tts.git
%cd kdext_conversa_ai_tts
```

> ⚠️ Replace `YOUR_USERNAME`. If the repo is **private**, use a personal access token:
> ```python
> !git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/kdext_conversa_ai_tts.git
> %cd kdext_conversa_ai_tts
> ```
> Generate a token at: [github.com/settings/tokens](https://github.com/settings/tokens) → **Fine-grained tokens** → select repo access.

### Step 7: Install dependencies (Cell 3)

```python
!pip install -q -r requirements.txt
```

⏳ This takes ~2-3 minutes (installs PyTorch, transformers, parler-tts, etc.)

---

## Part 4 — Verify Project

### Step 8: Check imports & config (Cell 4)

```python
import os
os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["WARMUP_ON_STARTUP"] = "false"
os.environ["DEVICE"] = "auto"
os.environ["TORCH_DTYPE"] = "float16"

from app.core.config import get_settings
from app.core.languages import INDIC_LANGUAGES, SUPPORTED_NON_INDIC_LANGUAGES

settings = get_settings()
print(f"✅ App name       : {settings.app_name}")
print(f"✅ Device config   : {settings.device}")
print(f"✅ Torch dtype     : {settings.torch_dtype}")
print(f"✅ Indic model     : {settings.indic_model_id}")
print(f"✅ Bark model      : {settings.bark_model_id}")
print(f"✅ Indic languages : {len(INDIC_LANGUAGES)}")
print(f"✅ Non-Indic langs : {len(SUPPORTED_NON_INDIC_LANGUAGES)}")
```

### Step 9: Run unit tests (Cell 5)

```python
!pytest tests/ -v --tb=short
```

✅ All tests should **pass** — they use mock models, no GPU needed.

---

## Part 5 — Load Models & Test TTS

### Step 10: Load Indic-Parler-TTS on GPU (Cell 6)

```python
import time, torch
from app.core.config import get_settings
from app.models.loaders import build_model_registry

get_settings.cache_clear()

settings = get_settings()
registry = build_model_registry(settings)

print("⏳ Loading Indic-Parler-TTS (first run downloads ~2-4 GB)...")
t0 = time.time()
indic_model = registry.get_model("indic_parler")
print(f"✅ Loaded in {time.time() - t0:.1f}s")
print(f"   Device : {indic_model.device}")
print(f"   Dtype  : {indic_model.torch_dtype}")

if torch.cuda.is_available():
    print(f"   GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

⏳ **First run**: 3-5 min (downloads model weights from HuggingFace).
⚡ **After that**: ~10-30 seconds (cached).

### Step 11: Test Hindi TTS (Cell 7)

```python
from IPython.display import Audio, display
from app.services.tts_service import TTSService
import wave
from io import BytesIO

service = TTSService(registry=registry, settings=settings)

print("🔊 Synthesizing Hindi...")
t0 = time.time()
wav_bytes = service.synthesize("नमस्ते, मेरा नाम कन्वर्सा है।", "hi")
print(f"✅ Done in {time.time() - t0:.2f}s ({len(wav_bytes)} bytes)")

with wave.open(BytesIO(wav_bytes), "rb") as wf:
    duration = wf.getnframes() / wf.getframerate()
    print(f"   Duration: {duration:.2f}s | Rate: {wf.getframerate()} Hz")

display(Audio(wav_bytes, rate=24000, autoplay=True))
```

🎧 You should **hear Hindi speech** directly in the Colab output!

### Step 12: Test Tamil (Cell 8)

```python
print("🔊 Synthesizing Tamil...")
t0 = time.time()
wav_ta = service.synthesize("வணக்கம், நான் கன்வர்சா.", "ta")
print(f"✅ Done in {time.time() - t0:.2f}s")
display(Audio(wav_ta, rate=24000, autoplay=True))
```

### Step 13: Test Bengali (Cell 9)

```python
print("🔊 Synthesizing Bengali...")
t0 = time.time()
wav_bn = service.synthesize("নমস্কার, আমি কনভার্সা।", "bn")
print(f"✅ Done in {time.time() - t0:.2f}s")
display(Audio(wav_bn, rate=24000, autoplay=True))
```

### Step 14: Load Bark & Test English (Cell 10)

```python
print("⏳ Loading Bark model...")
t0 = time.time()
bark_model = registry.get_model("bark")
print(f"✅ Bark loaded in {time.time() - t0:.1f}s")

print("\n🔊 Synthesizing English...")
t0 = time.time()
wav_en = service.synthesize(
    "Hello, this is a test of the Bark text to speech model.",
    "en",
    "v2/en_speaker_6",
)
print(f"✅ Done in {time.time() - t0:.2f}s")
display(Audio(wav_en, rate=24000, autoplay=True))
```

### Step 15: Test Custom Voice (Cell 11)

```python
print("🔊 Custom voice Hindi...")
t0 = time.time()
wav_custom = service.synthesize(
    "यह एक कस्टम वॉइस टेस्ट है।",
    "hi",
    "A soft, gentle female Indian voice with slow pacing and warm tone.",
)
print(f"✅ Done in {time.time() - t0:.2f}s")
display(Audio(wav_custom, rate=24000, autoplay=True))
```

---

## Part 6 — Test the FastAPI Server (Optional)

### Step 16: Start server & curl test (Cell 12)

```python
import subprocess, time

proc = subprocess.Popen(
    ["python", "run.py"],
    env={
        **os.environ,
        "LOAD_MODELS_ON_STARTUP": "true",
        "WARMUP_ON_STARTUP": "true",
        "DEVICE": "auto",
        "TORCH_DTYPE": "float16",
    },
)
time.sleep(60)  # wait for model loading
print("✅ Server running on port 8000")
```

### Step 17: Test API endpoints (Cell 13)

```python
# Health check
!curl -s http://localhost:8000/health | python -m json.tool

# Hindi TTS via API
!curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते","language":"hi"}' \
  --output /content/test_hindi.wav

# English TTS via API
!curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en"}' \
  --output /content/test_english.wav

print("✅ API test done!")
display(Audio("/content/test_hindi.wav", autoplay=True))
```

### Step 18: Stop server (Cell 14)

```python
proc.terminate()
print("🛑 Server stopped")
```

---

## Part 7 — GPU Memory Check

### Step 19: Memory summary (Cell 15)

```python
if torch.cuda.is_available():
    print("═" * 50)
    print("        GPU Memory Summary")
    print("═" * 50)
    print(f"  Allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved  : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Max alloc : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Device    : {torch.cuda.get_device_name(0)}")
    print("═" * 50)
```

---

## Quick Reference

| What | Expected Result |
|------|----------------|
| GPU check | `CUDA available: True`, `Tesla T4` |
| pip install | Completes without errors |
| pytest | All 5+ tests pass ✅ |
| Model load (first time) | 3-5 min download, then loads |
| Hindi synthesis | ~2-5s, plays audio ✅ |
| English synthesis | ~3-8s, plays audio ✅ |
| Health API | `{"status": "healthy", "model_loaded": true}` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA not available` | Runtime → Change runtime type → **GPU** |
| `ModuleNotFoundError` | Re-run `!pip install -r requirements.txt` |
| `OutOfMemoryError` | Use `suno/bark-small` (default), avoid loading both models simultaneously on free T4 |
| Slow first request | Normal — model downloads from HuggingFace on first load |
| Private repo clone fails | Use `https://TOKEN@github.com/...` format |
