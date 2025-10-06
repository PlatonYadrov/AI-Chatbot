from __future__ import annotations

from typing import Any, Dict, List

from ...shared.qdrant_client import get_qdrant


def search(vector: List[float], filters: Dict[str, Any] | None = None, top_k: int = 20, ef: int = 96) -> List[Dict[str, Any]]:
    client = get_qdrant()
    return client.search_dense("docs_ru", vector, top_k=top_k, ef=ef, filters=filters or {})


__all__ = ["search"]

