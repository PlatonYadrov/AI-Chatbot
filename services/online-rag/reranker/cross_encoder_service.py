from __future__ import annotations

from typing import Any, Dict, List


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    # Заглушка: оставляем первые top_k
    return candidates[:top_k]


__all__ = ["rerank"]

