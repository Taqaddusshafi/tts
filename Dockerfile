FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY run.py .
COPY scripts ./scripts
COPY .env.example .

RUN mkdir -p model_store

EXPOSE 8000

CMD ["python", "run.py"]

