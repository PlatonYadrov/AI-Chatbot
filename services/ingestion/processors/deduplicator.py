"""
Дедупликация: MinHash/SimHash + SHA256 (заглушка интерфейса).
Вход: normalized_docs; Выход: unique_docs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple


def compute_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def process(doc: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Возвращает (is_new, doc). Реальные MinHash/SimHash добавить позже."""
    doc = dict(doc)
    doc["hash"] = compute_sha256(doc.get("text", ""))
    # Заглушка: всегда считаем новым (без внешнего хранилища хэшей)
    return True, doc


__all__ = ["process", "compute_sha256"]

