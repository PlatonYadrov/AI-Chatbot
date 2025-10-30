# Online RAG Service (lightweight - reranking in Infinity container)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal system dependencies (no build-essential, gcc needed)
# torch/transformers are NOT installed - reranking happens in Infinity container
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (no torch!)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure static directory exists (for web UI)
RUN mkdir -p /app/static

EXPOSE 7000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7000/docs || exit 1

CMD ["uvicorn", "api.gateway:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
