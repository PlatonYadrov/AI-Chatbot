from __future__ import annotations

from typing import Any, Dict, List

from .dense_search import search as dense
from .sparse_search import search as sparse


def rrf_fusion(dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60, top_n: int = 100) -> List[Dict[str, Any]]:
    # Заглушка RRF: объединяем и обрезаем
    merged = (dense_results or []) + (sparse_results or [])
    return merged[:top_n]


def search(query: str, query_vector: List[float], filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    d = dense(query_vector, filters or {}, top_k=50, ef=96)
    s = sparse(query, filters or {}, top_k=50)
    return rrf_fusion(d, s)


__all__ = ["search"]

