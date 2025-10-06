from __future__ import annotations

from typing import Dict


def classify(query: str) -> Dict[str, str]:
    # Заглушка: всегда intent=general
    return {"intent": "general", "mode": "rag", "conf": "0.5"}


__all__ = ["classify"]

