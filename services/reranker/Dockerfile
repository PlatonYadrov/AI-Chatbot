# Custom Reranker Service with sentence-transformers
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app/

# Create cache directory
RUN mkdir -p /app/.cache

ENV HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

CMD ["python", "app.py"]

