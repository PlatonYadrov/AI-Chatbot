from __future__ import annotations

from typing import Any, Dict, List


def build(query: str, context: List[Dict[str, Any]]) -> str:
    # Заглушка строгого RAG-промпта
    citations = "\n".join([f"- {c.get('payload', {}).get('doc_id','')}" for c in context[:5]])
    return (
        "Ты помощник. Ответь кратко.\n"
        f"Вопрос: {query}\n"
        "Контекст доступен ниже.\n"
        f"Цитаты:\n{citations}\n"
    )


__all__ = ["build"]

