from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class QdrantClientStub:
    """Заглушка клиента Qdrant: держит коллекцию в памяти."""

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._storage: List[Dict[str, Any]] = []

    def upsert_batch(self, collection: str, points: List[Dict[str, Any]]) -> int:
        self._storage.extend(points)
        return len(points)

    def search_dense(self, collection: str, vector: List[float], top_k: int = 10, ef: int = 96, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Заглушка: возвращает первые top_k элементов
        return self._storage[:top_k]

    def alias_swap(self, old: str, new: str) -> None:
        return None


def get_qdrant() -> QdrantClientStub:
    return QdrantClientStub()


__all__ = ["QdrantClientStub", "get_qdrant"]

