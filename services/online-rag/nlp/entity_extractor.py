from __future__ import annotations

from typing import Any, Dict, List


def extract(query: str) -> Dict[str, Any]:
    # Заглушка NER: ничего не извлекаем
    return {"entities": [], "conf": "0.5"}


__all__ = ["extract"]

