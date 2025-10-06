"""
Indexer service: upsert в Qdrant (docs_ru dense; опц. docs_sparse).
Идемпотентность (point_id = sha256(doc_id:chunk_idx[:version])). Soft-delete устаревших версий.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def _point_id(doc_id: str, chunk_idx: int, version: str | None) -> str:
    raw = f"{doc_id}:{chunk_idx}:{version or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def upsert_batch(embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Заглушка: формирует payload для Qdrant upsert."""
    points = []
    for e in embeddings:
        pid = _point_id(e.get("doc_id", ""), int(e.get("chunk_idx", 0)), e.get("payload", {}).get("version"))
        points.append(
            {
                "id": pid,
                "vector": e.get("vector", []),
                "payload": e.get("payload", {}),
            }
        )
    # TODO: интеграция с shared.qdrant_client
    return points


__all__ = ["upsert_batch"]

