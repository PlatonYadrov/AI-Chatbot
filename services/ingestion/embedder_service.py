"""Client for local Text Embeddings Inference (TEI) server."""

from __future__ import annotations

import os
from typing import List, Dict, Any

import requests


TEI_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:8080")


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TEI_BASE_URL}{path}"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    payload = {"input": texts}
    data = _post("/embed", payload)
    # TEI returns { "data": [ {"embedding": [...], "index": 0}, ... ] }
    items = data.get("data", [])
    embeddings: List[List[float]] = [item.get("embedding", []) for item in items]
    return embeddings
