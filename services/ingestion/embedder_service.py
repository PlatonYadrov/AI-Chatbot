"""
Embedder service: HTTP /embed (sync) и/или consumer ready_for_embedding.
Батч-аккумулятор (до 32 текстов или timeout 5s). Redis L1 кэш (ключ emb:<sha1(text)>).
SLO: p95 ≤ 150ms/текст в батче. Заглушка без GPU.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def _fake_embed(texts: List[str]) -> List[List[float]]:
    # Заглушка: детерминированный вектор длины 8 на основе sha1
    vectors: List[List[float]] = []
    for t in texts:
        h = hashlib.sha1((t or "").encode("utf-8")).digest()
        vec = [int(x) / 255.0 for x in h[:8]]
        vectors.append(vec)
    return vectors


def embed_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [it.get("text", "") for it in items]
    vectors = _fake_embed(texts)
    result: List[Dict[str, Any]] = []
    for it, vec in zip(items, vectors):
        result.append(
            {
                "doc_id": it.get("doc_id"),
                "chunk_idx": it.get("chunk_idx"),
                "vector": vec,
                "payload": it.get("payload", {}),
            }
        )
    return result


__all__ = ["embed_batch"]

