Here are copy-paste ready commands for your demo:

---

## Local Demo (API only, no GPU needed)

**Terminal 1 — Start the server:**
```bash
cd ~/Desktop/tts
source .venv/bin/activate
LOAD_MODELS_ON_STARTUP=false WARMUP_ON_STARTUP=false python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Demo the endpoints:**
```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Show API handles validation correctly
curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "language": "zz"}' | python3 -m json.tool

# 3. Show error envelope for missing models
curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}' | python3 -m json.tool

# 4. Open interactive Swagger docs in browser
open http://localhost:8000/docs
```

---

## Full Demo on Google Colab (with actual TTS audio)

Paste these cells in order:

```python
# Cell 1 — Clone & install
!git clone https://github.com/YOUR_REPO/tts.git
%cd tts
!pip install -q -r requirements.txt
```

```python
# Cell 2 — Start server in background
import subprocess, time, os
os.environ["DEVICE"] = "auto"
os.environ["TORCH_DTYPE"] = "float16"
proc = subprocess.Popen(["python", "-m", "uvicorn", "app.main:app", "--port", "8000"])
time.sleep(30)  # wait for models to load
```

```python
# Cell 3 — Health check
!curl -s http://localhost:8000/health | python -m json.tool
```

```python
# Cell 4 — Generate Hindi speech
!curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, मैं एक AI हूँ", "language": "hi"}' \
  --output hindi_demo.wav

from IPython.display import Audio
Audio("hindi_demo.wav")
```

```python
# Cell 5 — Generate English speech
!curl -s -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a text to speech demo", "language": "en"}' \
  --output english_demo.wav

Audio("english_demo.wav")
```

---

> **Tip:** For the local demo, showing **Swagger UI** at `localhost:8000/docs` is the most impressive — it's interactive and lets you try endpoints live in the browser.