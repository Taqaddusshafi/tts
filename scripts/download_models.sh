#!/usr/bin/env bash
set -euo pipefail

MODEL_STORE_DIR="${MODEL_STORE_DIR:-model_store}"
INDIC_MODEL_ID="${INDIC_MODEL_ID:-ai4bharat/indic-parler-tts}"
BARK_MODEL_ID="${BARK_MODEL_ID:-suno/bark-small}"

mkdir -p "${MODEL_STORE_DIR}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_store_dir = os.environ.get("MODEL_STORE_DIR", "model_store")
models = [
    os.environ.get("INDIC_MODEL_ID", "ai4bharat/indic-parler-tts"),
    os.environ.get("BARK_MODEL_ID", "suno/bark-small"),
]

for model_id in models:
    print(f"Downloading {model_id} into {model_store_dir}")
    snapshot_download(repo_id=model_id, cache_dir=model_store_dir)

print("Model download complete.")
PY

