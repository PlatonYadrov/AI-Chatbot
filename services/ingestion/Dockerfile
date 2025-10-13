FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCLING_OFFLINE_MODE=true \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/ \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Optional: Download models during build (uncomment for air-gapped deployment)
# RUN python download_models.py || echo "Models will download on first run"

EXPOSE 6000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "6000"]
