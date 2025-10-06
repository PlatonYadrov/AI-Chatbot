from __future__ import annotations

from typing import Any, Dict, List


def generate(query: str, context: List[Dict[str, Any]], mode: str = "rag") -> Dict[str, Any]:
    # Заглушка инференса
    text = f"(stub) Ответ на '{query}' с {len(context)} контекстами"
    return {"text": text, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


__all__ = ["generate"]

